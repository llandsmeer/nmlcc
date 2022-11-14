/*
 * to_bytecode
 * const void * get_catalogue(int * n) {
 *    *n = NUM_MECHS;
 *    static arb_mechanism cat[NUM_MECHS] = { make_mech(), mak .. }
 *    return (void*)cat;
 *
 *    type
 *    i_cpu =() -> arb_mechanism_inferface *
 *          partition_width = 1
 *          backend = arb_backend_kind_cpu
 *          alignment = 8
 *          init_mechanism = (pp*) => void
 *          compute_current = (pp*) => void
 *          apply_events = (pp*, stream*) => void
 *          advance_state = (pp*) => void
 *          write_ions = (pp*) => void
 *          post_events = (pp*) => void
 *    i_gpu = () => null
 *
 *    Ok this all works?
 *    'Just' need to implement it in rust
 *    Or I write it in CPP, make the 'bytecode' an argument or something
 *    Or cranelift? No onyl amd64&aarch64
 *    Life sucks
 *    Ok but the only thing that makes sense then is a javascript backend? We could even use
 *    that in python (o.o the horror)
 *    Oh we were doing that already
 *    So we only need init/compute/apply etc
 *    Output asm.js?
 *    Bytecode also still sounds very easy. Esp since it's fully typed
 *
 *
 *    somewhere in the print_dependency_chains things
 * }
 * */


use crate::{
    expr::{Op, Cmp, Expr, Boolean, Stmnt, Quantity},
    nmodl::{Nmodl, sorted_dependencies_of, find_dependencies},
    error::{Error, Result},
    Map
};

macro_rules! err {
    ($($arg:tt)*) => {{
        return Err(Error::Bytecode { what: format!($($arg)*) })
    }};
}

#[derive(Debug)]
pub enum Bytecode {
    ReserveLocals(usize),
    StoreLocal(usize),
    LoadLocal(usize),
    StoreState(usize), // STILL NEED TO SEPARATE GLOBALS FROM PARAMS!
    IntegrateState(usize),
    StoreCurrent(),
    StoreIonCurrent(usize),
    LoadState(usize),
    LoadParameter(usize),
    PushConstant(f64),
    LoadV(),
    LoadIonIntConc(usize),
    LoadIonOutConc(usize),
    LoadIonErev(usize),
    LoadIonCurrent(usize),
    Add(usize),
    Mul(usize),
    Pow(usize),
    Exp(),
    Log(),
    Sqrt(),
    BoolAnd(),
    BoolOr(),
    BoolEq(),
    BoolNe(),
    BoolLe(),
    BoolGe(),
    BoolLt(),
    BoolGt(),
    PushTrue(),
    PushFalse(),
    Nop(),
    JmpRel(usize), /* s.t. JmpRel(0) is a nop instruction -> we can never have loops */
    JmpRelIfFalse(usize),
    Done()
}

#[derive(Clone)]
struct CompilationContext {
    locals: Map<String, usize>,
    params: Map<String, usize>,
    state: Map<String, usize>,
    ions: Map<String, usize>
}

fn build_names_map(names: &[String]) -> Map<String, usize> {
    names.iter().enumerate().map(|(i, name)| (name.to_string(), i)).collect()
}

impl CompilationContext {
    fn new() -> Self {
        Self {
            locals: Map::new(),
            params: Map::new(),
            state: Map::new(),
            ions: Map::new()
        }
    }

    fn with_locals(self, locals: &[String]) -> Self {
        Self {
            locals: build_names_map(locals),
            ..self
        }
    }

    fn with_params(self, params: &[String]) -> Self {
        Self {
            params: build_names_map(params),
            ..self
        }
    }

    fn with_state(self, state: &[String]) -> Self {
        Self {
            state: build_names_map(state),
            ..self
        }
    }

    fn with_ions(self, state: &[String]) -> Self {
        Self {
            ions: build_names_map(state),
            ..self
        }
    }

    pub fn from_nmodl(n: &Nmodl) -> Self {
        CompilationContext::new()
            .with_state(&n.state.iter().map(|s| s.clone()).collect::<Vec<_>>())
            .with_params(&n.parameters.keys().map(|s| s.clone()).collect::<Vec<_>>())
            .with_ions(&[
                       n.known_ions.iter().map(|s| s.clone()).collect::<Vec<_>>(),
                       n.species.clone()
            ].concat().iter().filter(|s| !s.is_empty()).map(|s| s.clone()).collect::<Vec<_>>())
    }

    pub fn compile_stmnt(&self, out: &mut Vec<Bytecode>, s: &Stmnt) -> Result<()> {
        use Bytecode::*;
        Ok(match s {
            Stmnt::Ass(name, e) => {
                self.compile_expr(out, e)?;
                if let Some(i) = self.locals.get(name) {
                    out.push(StoreLocal(*i));
                } else if let Some(i) = self.state.get(name) {
                    out.push(StoreState(*i))
                } else if name.ends_with("'") && self.state.contains_key(&name[..name.len()-1])  {
                    out.push(IntegrateState(self.state[&name[..name.len()-1]]));
                } else if name == "i" {
                    out.push(StoreCurrent())
                } else if name.starts_with("i") && self.ions.contains_key(&name[1..])  {
                    out.push(StoreIonCurrent(self.ions[&name[1..]]))
                } else {
                    err!("can not store to name {name:?}")
                }
            }
            Stmnt::Ift(c, a, b) => {
                /* MAYBE SHOULD MAKE THIS BRANCHLESS! FOR SIMD */
                self.compile_bool(out, c)?;
                let ref_jump_to_b = out.len();
                out.push(Nop()); /* if-branch to be */
                self.compile_stmnt(out, a)?;
                if let Some(b) = b.as_ref() {
                    let ref_jump_over_b = out.len();
                    out.push(Nop()); /* jump-over else to be */
                    out[ref_jump_to_b] = JmpRelIfFalse(out.len() - ref_jump_to_b - 1);
                    self.compile_stmnt(out, b)?;
                    out[ref_jump_over_b] = JmpRel(out.len() - ref_jump_over_b - 1)
                } else {
                    out[ref_jump_to_b] = JmpRelIfFalse(out.len() - ref_jump_to_b - 1);
                }
            }
        })
    }

    fn compile_block(&self, n: &Nmodl, block: &Map<String, Stmnt>) -> Result<Vec<Bytecode>> {
        let roots = &block.keys().cloned().collect::<Vec<String>>();
        let vars = &n
            .variables
            .iter()
            .chain(block.iter())
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect();
        let dependencies = find_dependencies(vars);
        let locals = sorted_dependencies_of(roots, &dependencies, &n.symbols)?;
        let mut result = Vec::new();
        let ctx = self.clone().with_locals(&locals);
        result.push(Bytecode::ReserveLocals(locals.len()));
        for d in locals {
            if let Some(s) = vars.get(&d) {
                let s = fold_constants(s, &n.constants);
                ctx.compile_stmnt(&mut result, &s)?;
            } else {
                err!("unresolved variable {d}");
            }
        }
        for s in block.values() {
            let s = fold_constants(s, &n.constants);
            ctx.compile_stmnt(&mut result, &s)?;
        }
        result.push(Bytecode::Done());
        Ok(result)
    }

    pub fn compile_bool(&self, out: &mut Vec<Bytecode>, b: &Boolean) -> Result<()> {
        use Bytecode::*;
        macro_rules! op {
            ($bytecode: tt, $x: expr, $y: expr) => {
                {
                    self.compile_bool(out, $x)?;
                    self.compile_bool(out, $y)?;
                    out.push($bytecode());
                }
            }
        }
        macro_rules! cmp {
            ($bytecode: tt, $x: expr, $y: expr) => {
                {
                    self.compile_expr(out, $x)?;
                    self.compile_expr(out, $y)?;
                    out.push($bytecode());
                }
            }
        }
        Ok(match b {
            Boolean::Op(Op::And, x, y) => op!(BoolAnd, x, y),
            Boolean::Op(Op::Or, x, y) => op!(BoolOr, x, y),
            Boolean::Cmp(Cmp::Eq, x, y) => cmp!(BoolEq, x, y),
            Boolean::Cmp(Cmp::Ne, x, y) => cmp!(BoolNe, x, y),
            Boolean::Cmp(Cmp::Le, x, y) => cmp!(BoolLe, x, y),
            Boolean::Cmp(Cmp::Ge, x, y) => cmp!(BoolGe, x, y),
            Boolean::Cmp(Cmp::Lt, x, y) => cmp!(BoolLt, x, y),
            Boolean::Cmp(Cmp::Gt, x, y) => cmp!(BoolGt, x, y),
            Boolean::Lit(b) => out.push(if *b { PushTrue() } else { PushFalse() }),
        })
    }

    pub fn compile_expr(&self, out: &mut Vec<Bytecode>, e: &Expr) -> Result<()> {
        use Bytecode::*;
        macro_rules! binop {
            ($bytecode: tt, $es: expr) => {
                {
                    let es = $es;
                    for e in es {
                        self.compile_expr(out, e)?;
                    }
                    out.push($bytecode(es.len()));
                }
            }
        }
        macro_rules! fun1 {
            ($bytecode: tt, $e: expr) => {
                {
                    {
                        self.compile_expr(out, $e)?;
                        out.push($bytecode());
                    }
                }
            }
        }
        Ok(match e {
            Expr::F64(x) => out.push(PushConstant(*x)),
            Expr::Var(name) => out.push(
                if let Some(i) = self.locals.get(name) {
                    LoadLocal(*i)
                } else if let Some(i) = self.state.get(name) {
                    LoadState(*i)
                } else if let Some(i) = self.params.get(name) {
                    LoadParameter(*i)
                } else if "v" == name {
                    LoadV()
                } else if name == ""{
                    err!("can not load empty name")
                } else if name.ends_with("i") && self.ions.contains_key(&name[..name.len()-1])  {
                    LoadIonIntConc(self.ions[&name[..name.len()-1]])
                } else if name.ends_with("o") && self.ions.contains_key(&name[..name.len()-1])  {
                    LoadIonOutConc(self.ions[&name[..name.len()-1]])
                } else if name.starts_with("e") && self.ions.contains_key(&name[1..])  {
                    LoadIonErev(self.ions[&name[1..]])
                } else if name.starts_with("i") && self.ions.contains_key(&name[1..])  {
                    LoadIonCurrent(self.ions[&name[1..]])
                } else {
                    err!("can not load from name {name}")
                }
            ),
            Expr::Add(es) => binop!(Add, es),
            Expr::Mul(es) => binop!(Mul, es),
            Expr::Pow(es) => binop!(Pow, es),
            Expr::Exp(x) => fun1!(Exp, x.as_ref()),
            Expr::Log(x) => fun1!(Log, x.as_ref()),
            Expr::Sqrt(x) => fun1!(Sqrt, x.as_ref()),
            Expr::ProximalDistanceFromRegion(_) => err!("ProximalDistanceFromRegion not in bytecode"),
            Expr::DistanceFromRoot() => err!("DistanceFromRoot not in bytecode"),
            Expr::Fun(name, _x) => {
                match name {
                    _ => { err!("Function {name} not supported in bytecode") }
                }
            }
        })
    }
}

fn fold_constants(s: &Stmnt, constants: &Map<String, Quantity>) -> Stmnt {
    s.map(&|e: &Expr| -> Expr{
        if let Expr::Var(x) = e {
            if let Some(q) = constants.get(x) {
                return Expr::F64(q.value)
            }
        }
        e.clone()
    })
}

pub fn mk_bytecode(n: Nmodl) -> Result<String> {
    let n = &n.simplify();
    if !n.transitions.is_empty() {
        err!("bytecode doesn't support transitions because I have no idea what that is");
    }
    let ctx = CompilationContext::from_nmodl(&n);
    let init_mechanism = ctx.compile_block(&n, &n.init)?;
    let mut update = n.deriv.clone();
    println!("{:?}" ,&n.species);
    for (k, v) in n.outputs.iter() {
        if update.contains_key(k) {
            err!("breakpoint overwrites deriv {k}");
        }
        update.insert(k.to_string(), v.clone());
    }
    let compute_currents = ctx.compile_block(&n, &update)?;
    Ok(compute_currents.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>().join("\n"))
}


#[test]
fn bytecode_simple() {
    use crate::expr::parse::*;
    fn run(code: &[Bytecode]) -> f64 {
        let mut stack: Vec<f64> = Vec::new();
        let mut locals: Vec<f64> = Vec::new();
        let mut i = 0;
        while i < code.len() {
            eprintln!("Exec {:?} \t\tstack = {:?}\t\t locals={:?}", code[i], stack, locals);
            match code[i] {
                Bytecode::PushConstant(x) => stack.push(x),
                Bytecode::Add(n) => {
                    let mut r = 0.0;
                    for _ in 0..n {
                        r += stack.pop().unwrap();
                    }
                    stack.push(r);
                },
                Bytecode::ReserveLocals(n) => locals = vec![0.0; n],
                Bytecode::LoadLocal(n) => stack.push(locals[n]),
                Bytecode::PushTrue() => stack.push(1.0),
                Bytecode::PushFalse() => stack.push(0.0),
                Bytecode::JmpRelIfFalse(n) => {
                    if stack.pop().unwrap() == 0.0 {
                        i += n
                    }
                },
                Bytecode::StoreLocal(n) => locals[n] = stack.pop().unwrap(),
                Bytecode::JmpRel(n) => i += n,
                Bytecode::BoolGt() => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(if a > b { 1.0 } else { 0.0 })
                }
                _ => todo!("unknown op {:?}", code[i])
            }
            eprintln!("       => \t\tstack = {:?}\t\t locals={:?}", stack, locals);
            i += 1;
        }
        if stack.len() == 1 {
            stack[0]
        } else {
            panic!("stack corruption");
        }
    }
    //////////////////////
    let mut result = Vec::new();
    CompilationContext::new().compile_expr(&mut result, &expr("1 + 1").unwrap().1).unwrap();
    assert_eq!(run(&result), 2.0);
    //////////////////////
    let mut result = Vec::new();
    result.push(Bytecode::ReserveLocals(1));
    CompilationContext::new()
        .with_locals(&[String::from("x")])
        .compile_stmnt(&mut result,
                           &Stmnt::Ift(
                               boolean("1 .gt. 0").unwrap().1,
                               Box::new(Stmnt::Ass("x".to_string(), expr("1 + 1").unwrap().1)),
                               Box::new(Some(Stmnt::Ass("x".to_string(), expr("2 + 2").unwrap().1)))
                           )).unwrap();
    result.push(Bytecode::LoadLocal(0));
    assert_eq!(run(&result), 2.0);
    //////////////////////
    let mut result = Vec::new();
    result.push(Bytecode::ReserveLocals(1));
    CompilationContext::new()
        .with_locals(&[String::from("x")])
        .compile_stmnt(&mut result,
                           &Stmnt::Ift(
                               boolean("1 .gt. 3").unwrap().1,
                               Box::new(Stmnt::Ass("x".to_string(), expr("1 + 1").unwrap().1)),
                               Box::new(Some(Stmnt::Ass("x".to_string(), expr("2 + 2").unwrap().1)))
                           )).unwrap();
    result.push(Bytecode::LoadLocal(0));
    assert_eq!(run(&result), 4.0);
}
