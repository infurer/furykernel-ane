// SPDX-FileCopyrightText: 2026 Mika Tammi
// SPDX-License-Identifier: BSD-3-Clause

//! The Neural Engine conformance run: every case of the corpus `Cases.hs`
//! writes, compiled by Core ML as a runtime compiles it, where its ops run
//! asserted (`MLComputePlan`), predicted twice, and its outputs held to the
//! core's evaluator's, the stage's meaning, and to the interpreter's, the
//! lowered program's, both in f64. Run as `nix run .#ane-conformance`, or
//! on a corpus of one's own as `nix run .#ane-conformance -- CORPUS`.
//!
//! A case is held to:
//!
//! - on the Neural Engine (`.cpuAndNeuralEngine`), every op the plan gives a
//!   device placed on it, but `reduce_argmax`, which Core ML keeps on the
//!   CPU; an op the plan gives no device is fused into another;
//! - two predictions of the same bits;
//! - each float output's rows (its elements at one position of the last
//!   axis) within its bound of the row's largest magnitude in each
//!   reference, or of [`ABSOLUTE`] if that is larger, and finite, so an
//!   all-zero row exactly: on the Neural Engine [`TAU`], or [`TAU_SOFTCAP`]
//!   for an output made through a softcap (`softcap`), and on the CPU
//!   [`TAU_CPU`]; the interpreter's model of the Neural Engine is reported
//!   beside them, against the evaluator, and held to nothing;
//! - an output that pins what the Neural Engine was observed to compute
//!   (`model`) equal to the interpreter's model of it, bit for bit;
//! - each token the first largest of the logits the prediction made, and of
//!   a reference logit within twice its logits' bound of the row's
//!   largest; a token of logits the corpus makes tie (`exact`), the
//!   evaluator's.
//!
//! The bounds are checked against the corpus rather than derived; there
//! the Neural Engine's worst row is within 0.54 of its bound, a softcap's
//! within 0.40 of its own (`CLAUDE.md` gives the measurements).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::mpsc;
use std::{env, fs, process};

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{AnyThread, ClassType};
use objc2_core_ml::{
    MLCPUComputeDevice, MLComputeDeviceProtocol, MLComputePlan, MLComputeUnits,
    MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLGPUComputeDevice, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType, MLNeuralEngineComputeDevice,
};
use objc2_foundation::{
    NSArray, NSDictionary, NSError, NSInteger, NSNumber, NSObjectProtocol, NSProcessInfo, NSString,
    NSURL,
};

/// How far a float output's row may be from a reference's on the Neural
/// Engine: this much of the row's largest magnitude there, 2^-7, 16 times
/// binary16's unit roundoff.
const TAU: f64 = 1.0 / 128.0;

/// How far an output made through a softcap may be: 2^-6, as the Neural
/// Engine's tanh is up to 0.58 % low for small arguments.
const TAU_SOFTCAP: f64 = 1.0 / 64.0;

/// What a row of small magnitude is held to absolutely on the Neural
/// Engine, 2^-9: its bound times the larger of its largest magnitude and
/// `ABSOLUTE / τ` (2^-2 for [`TAU`], 2^-3 for [`TAU_SOFTCAP`]). The Neural
/// Engine's sums err by about 2^-16·√(k/12) over a fan-in of k, whatever
/// the size of their outputs.
const ABSOLUTE: f64 = 1.0 / 512.0;

/// What a run on the CPU is held to, any output: 2^-5 of a row's largest
/// magnitude or of 2^-3. Core ML's CPU sums in binary16 (its convolutions
/// are four to eight times as far from f64 as the Neural Engine's, and
/// overflow past 65504).
const TAU_CPU: f64 = 1.0 / 32.0;

/// The least magnitude a row on the CPU is held relative to.
const FLOOR_CPU: f64 = 0.125;

/// The ops Core ML runs on the CPU at the end of a program it places on the
/// Neural Engine: an argmax, and the concat that joins an output layer's
/// chunks for it or for an output.
const TAIL: &[&str] = &["reduce_argmax", "concat"];

/// The compute units a case runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Units {
    /// The CPU and the Neural Engine, every op but the [`TAIL`]'s on the latter.
    NeuralEngine,
    /// The CPU alone.
    Cpu,
}

impl Units {
    fn parse(s: &str) -> Units {
        match s {
            "ane" => Units::NeuralEngine,
            "cpu" => Units::Cpu,
            _ => fail(&format!("compute units {s}")),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Units::NeuralEngine => "ane",
            Units::Cpu => "cpu",
        }
    }

    /// How far a float output's row may be from a reference, held this
    /// way, and the least magnitude it is held relative to.
    fn bound(self, hold: Hold) -> (f64, f64) {
        match (self, hold) {
            (Units::NeuralEngine, Hold::Softcap) => (TAU_SOFTCAP, ABSOLUTE / TAU_SOFTCAP),
            (Units::NeuralEngine, _) => (TAU, ABSOLUTE / TAU),
            (Units::Cpu, _) => (TAU_CPU, FLOOR_CPU),
        }
    }

    fn compute_units(self) -> MLComputeUnits {
        match self {
            Units::NeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
            Units::Cpu => MLComputeUnits::CPUOnly,
        }
    }
}

/// How a token output is held.
#[derive(Debug, PartialEq)]
enum Token {
    /// It is taken from this output's logits.
    Logits(String),
    /// Its logits tie as the corpus made them: the evaluator's.
    Exact,
}

/// How a float output is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hold {
    /// Within its bound of each reference.
    Plain,
    /// Within the bound of an output made through a softcap.
    Softcap,
    /// To the interpreter's model of the Neural Engine, bit for bit.
    Model,
}

/// What an output holds.
#[derive(Debug, PartialEq)]
enum Kind {
    Float(Hold),
    Token(Token),
}

#[derive(Debug, PartialEq)]
struct Output {
    name: String,
    kind: Kind,
    shape: Vec<usize>,
}

impl Output {
    /// Its rows: positions of its last axis.
    fn rows(&self) -> usize {
        self.shape.last().copied().unwrap_or(1)
    }
}

struct Case {
    name: String,
    dir: PathBuf,
    inputs: Vec<(String, Vec<usize>)>,
    outputs: Vec<Output>,
    units: Vec<Units>,
}

fn fail(what: &str) -> ! {
    eprintln!("ane-conformance: {what}");
    process::exit(2)
}

fn read(p: &Path) -> Vec<u8> {
    fs::read(p).unwrap_or_else(|e| fail(&format!("{}: {e}", p.display())))
}

fn read_text(p: &Path) -> String {
    String::from_utf8(read(p)).unwrap_or_else(|_| fail(&format!("{}: not UTF-8", p.display())))
}

fn read_f64(p: &Path) -> Vec<f64> {
    read(p)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect()
}

fn read_f16(p: &Path) -> Vec<u16> {
    read(p)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}

fn parse_shape(s: &str) -> Vec<usize> {
    s.split(',')
        .map(|d| {
            d.parse()
                .unwrap_or_else(|_| fail(&format!("a dimension {d}")))
        })
        .collect()
}

fn parse_output(line: &str) -> Output {
    let words: Vec<&str> = line.split_whitespace().collect();
    let (name, kind, shape) = match words.as_slice() {
        [name, "float", shape] => (name, Kind::Float(Hold::Plain), shape),
        [name, "float", shape, "softcap"] => (name, Kind::Float(Hold::Softcap), shape),
        [name, "float", shape, "model"] => (name, Kind::Float(Hold::Model), shape),
        [name, "token", shape, "exact"] => (name, Kind::Token(Token::Exact), shape),
        [name, "token", shape, logits] => (
            name,
            Kind::Token(Token::Logits((*logits).to_owned())),
            shape,
        ),
        _ => fail(&format!("an output line {line}")),
    };
    Output {
        name: (*name).to_owned(),
        kind,
        shape: parse_shape(shape),
    }
}

fn parse_case(dir: &Path) -> Case {
    let lines = |f: &str| -> Vec<String> {
        read_text(&dir.join(f))
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(str::to_owned)
            .collect()
    };
    let inputs = lines("inputs.txt")
        .iter()
        .map(
            |l| match l.split_whitespace().collect::<Vec<_>>().as_slice() {
                [name, shape] => ((*name).to_owned(), parse_shape(shape)),
                _ => fail(&format!("an input line {l}")),
            },
        )
        .collect();
    Case {
        name: dir
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
        dir: dir.to_owned(),
        inputs,
        outputs: lines("outputs.txt")
            .iter()
            .map(String::as_str)
            .map(parse_output)
            .collect(),
        units: lines("units.txt")
            .iter()
            .map(|l| Units::parse(l.trim()))
            .collect(),
    }
}

/// A binary16 number's value.
fn half_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((h >> 10) & 0x1f);
    let fraction = f64::from(h & 0x3ff);
    match exponent {
        0 => sign * fraction * 2f64.powi(-24),
        31 if fraction == 0.0 => sign * f64::INFINITY,
        31 => f64::NAN,
        _ => sign * (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15),
    }
}

/// Element `i` of an array of this shape in C order, as an offset by these strides.
fn offset(shape: &[usize], strides: &[usize], mut i: usize) -> usize {
    let mut at = 0;
    for (d, s) in shape.iter().zip(strides).rev() {
        at += (i % d) * s;
        i /= d;
    }
    at
}

fn strides_of(a: &NSArray<NSNumber>) -> Vec<usize> {
    a.iter()
        .map(|n| usize::try_from(n.integerValue()).unwrap_or(0))
        .collect()
}

/// The worst row of a float output against a reference: its largest error
/// over the row's largest magnitude in the reference, or over `floor` if
/// that is larger, and the row; a value not finite is infinitely far.
fn float_error(got: &[f64], want: &[f64], rows: usize, floor: f64) -> (f64, usize) {
    let mut worst = (0.0, 0);
    for r in 0..rows {
        let (mut err, mut scale) = (0.0_f64, 0.0_f64);
        for (g, w) in got.iter().zip(want).skip(r).step_by(rows) {
            err = err.max(if g.is_finite() {
                (g - w).abs()
            } else {
                f64::INFINITY
            });
            scale = scale.max(w.abs());
        }
        let ratio = err / scale.max(floor);
        if ratio > worst.0 || ratio.is_nan() {
            worst = (ratio, r);
        }
    }
    worst
}

/// A token's index, if it is one.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn index(t: f64) -> Option<usize> {
    // An integer from 0 below 2^53 converts exactly.
    if t >= 0.0 && t.fract() == 0.0 && t < 2f64.powi(53) {
        Some(t as usize)
    } else {
        None
    }
}

/// The first largest of a row of logits, `rows` the logits' last axis.
fn first_largest(logits: &[f64], row: usize, rows: usize) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (k, &x) in logits.iter().skip(row).step_by(rows).enumerate() {
        if best.is_none_or(|(_, b)| x > b) {
            best = Some((k, x));
        }
    }
    best.map(|(k, _)| k)
}

/// How many of an output's elements differ from the model's, and the first
/// of them; equal values are equal whatever their signs of zero, and NaNs
/// equal.
fn model_errors(got: &[f64], model: &[f64]) -> (usize, Option<usize>) {
    // Exact equality is the rule, but for zeros' signs and NaNs' payloads.
    #[allow(clippy::float_cmp)]
    let differ = |(g, m): (&f64, &f64)| !(g == m || (g.is_nan() && m.is_nan()));
    let count =
        got.iter().zip(model).filter(|&p| differ(p)).count() + got.len().abs_diff(model.len());
    let first = got.iter().zip(model).position(differ);
    (count, first)
}

/// The rows whose tokens break the rules, against the evaluator's tokens
/// and, for tokens of logits, the predicted and the reference logits,
/// whose bound is `tau`.
fn token_errors(
    rule: &Token,
    got: &[f64],
    want: &[f64],
    logits: Option<(&[f64], &[f64])>,
    tau: f64,
) -> Vec<String> {
    let rows = got.len();
    let mut errors = Vec::new();
    for r in 0..rows {
        let Some(t) = index(got[r]) else {
            errors.push(format!("row {r}: {} is no token", got[r]));
            continue;
        };
        match (rule, logits) {
            (Token::Exact, _) => {
                if index(want[r]) != Some(t) {
                    errors.push(format!("row {r}: {t}, the evaluator's {}", want[r]));
                }
            }
            (Token::Logits(_), Some((predicted, reference))) => {
                if first_largest(predicted, r, rows) != Some(t) {
                    errors.push(format!(
                        "row {r}: {t}, not the first largest of its logits ({:?})",
                        first_largest(predicted, r, rows)
                    ));
                }
                let row: Vec<f64> = reference.iter().skip(r).step_by(rows).copied().collect();
                let largest = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let scale = row.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                match row.get(t) {
                    Some(&x) if x >= largest - 2.0 * tau * scale => {}
                    x => errors.push(format!(
                        "row {r}: {t}, its reference logit {x:?} below the largest, {largest}, by more than 2τ ({tau}) of {scale}"
                    )),
                }
            }
            (Token::Logits(n), None) => errors.push(format!("no logits {n}")),
        }
    }
    errors
}

fn file_url(p: &Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&p.to_string_lossy()))
}

fn described(e: &NSError) -> String {
    e.localizedDescription().to_string()
}

fn configuration(units: Units) -> Retained<MLModelConfiguration> {
    // SAFETY: a plain configuration object, set before use.
    let c = unsafe { MLModelConfiguration::new() };
    // SAFETY: as above.
    unsafe { c.setComputeUnits(units.compute_units()) };
    c
}

/// A directory copied whole, links followed, its files writable: Core ML
/// compiles no package whose files are links, as a store path joining
/// others has them.
fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let path = entry?.path();
        let dest = to.join(path.file_name().unwrap_or_default());
        if path.is_dir() {
            copy_dir(&path, &dest)?;
        } else {
            fs::copy(&path, &dest)?;
            fs::set_permissions(&dest, fs::Permissions::from_mode(0o644))?;
        }
    }
    Ok(())
}

/// The package compiled by Core ML, as a runtime compiles it, from a copy
/// of its own: into a directory of Core ML's.
fn compile(case: &Case) -> Result<PathBuf, String> {
    let copy = env::temp_dir().join(format!(
        "furykernel-ane-conformance-{}-{}",
        process::id(),
        case.name
    ));
    let package = copy.join("model.mlpackage");
    copy_dir(&case.dir.join("model.mlpackage"), &package)
        .map_err(|e| format!("copying the package: {e}"))?;
    #[allow(deprecated)]
    // SAFETY: a file URL of a package copied from the corpus.
    let compiled = unsafe { MLModel::compileModelAtURL_error(&file_url(&package)) };
    fs::remove_dir_all(&copy).ok();
    let url = compiled.map_err(|e| format!("compiling: {}", described(&e)))?;
    url.path()
        .map(|p| PathBuf::from(p.to_string()))
        .ok_or_else(|| "a compiled model of no path".to_owned())
}

/// A thing moved across threads: Core ML's objects are thread safe, the
/// bindings don't say so.
struct Sendable<T>(T);
// SAFETY: only Core ML's compute plan and an error's text cross from Core
// ML's completion handler to the caller.
unsafe impl<T> Send for Sendable<T> {}

/// An op of a compiled model, as Core ML's plan places it: its operator,
/// without its opset's prefix (`ios18.`), its device (`ane`, `gpu`, `cpu`,
/// or `fused` for an op the plan gives none, fused into the op that makes
/// its input), and the values it reads and makes.
struct Planned {
    operator: String,
    device: &'static str,
    reads: Vec<String>,
    makes: Vec<String>,
}

/// The ops placed against the rule: off the Neural Engine, unless fused
/// or of the program's tail, a [`TAIL`] op whose values no op on the
/// Neural Engine reads.
fn misplaced(ops: &[Planned]) -> Vec<String> {
    let mut on_ane: BTreeMap<&str, bool> = BTreeMap::new();
    for op in ops {
        for r in &op.reads {
            *on_ane.entry(r.as_str()).or_default() |= op.device == "ane";
        }
    }
    ops.iter()
        .filter(|op| op.device != "ane" && op.device != "fused")
        .filter(|op| {
            !(TAIL.contains(&op.operator.as_str())
                && op
                    .makes
                    .iter()
                    .all(|m| !on_ane.get(m.as_str()).copied().unwrap_or(false)))
        })
        .map(|op| {
            format!(
                "{} ({}) on the {}",
                op.operator,
                op.makes.join(","),
                op.device
            )
        })
        .collect()
}

fn placement(compiled: &Path, units: Units) -> Result<Vec<Planned>, String> {
    let (tx, rx) = mpsc::channel::<Sendable<Result<Retained<MLComputePlan>, String>>>();
    let block = block2::RcBlock::new(move |plan: *mut MLComputePlan, err: *mut NSError| {
        // SAFETY: Core ML hands the handler a plan or an error, each valid
        // for the call; retaining keeps it past it.
        let plan = unsafe { Retained::retain(plan) };
        let result = plan.ok_or_else(|| {
            // SAFETY: as above.
            unsafe { Retained::retain(err) }.map_or_else(|| "no plan".to_owned(), |e| described(&e))
        });
        tx.send(Sendable(result)).ok();
    });
    // SAFETY: a file URL of a compiled model, a configuration, and a
    // handler that outlives the call (Core ML copies it).
    unsafe {
        MLComputePlan::loadContentsOfURL_configuration_completionHandler(
            &file_url(compiled),
            &configuration(units),
            &block,
        );
    }
    let plan = rx
        .recv()
        .map_err(|e| e.to_string())?
        .0
        .map_err(|e| format!("planning: {e}"))?;
    // SAFETY: the structure of a loaded plan.
    let structure = unsafe { plan.modelStructure() };
    // SAFETY: as above.
    let program = unsafe { structure.program() }.ok_or("not an ML program")?;
    // SAFETY: as above.
    let functions = unsafe { program.functions() };
    let main = functions
        .objectForKey(&NSString::from_str("main"))
        .ok_or("no main function")?;
    let mut planned = Vec::new();
    // SAFETY: the main function's block, of the plan's structure.
    for op in unsafe { main.block().operations() } {
        // SAFETY: an operation of the plan's structure.
        let name = unsafe { op.operatorName() }.to_string();
        let operator = name
            .split_once('.')
            .map_or(name.as_str(), |(_, o)| o)
            .to_owned();
        if operator == "const" {
            continue;
        }
        // SAFETY: as above.
        let device = match unsafe { plan.computeDeviceUsageForMLProgramOperation(&op) } {
            Some(usage) => {
                // SAFETY: the usage's preferred device.
                let preferred = unsafe { usage.preferredComputeDevice() };
                kind(&preferred)
            }
            None => "fused",
        };
        // SAFETY: as above.
        let makes = unsafe { op.outputs() }
            .iter()
            // SAFETY: an output of the operation.
            .map(|o| unsafe { o.name() }.to_string())
            .collect();
        let mut reads = Vec::new();
        // SAFETY: as above.
        let inputs = unsafe { op.inputs() };
        for key in inputs.allKeys() {
            let Some(argument) = inputs.objectForKey(&key) else {
                continue;
            };
            // SAFETY: an argument of the operation.
            for binding in unsafe { argument.bindings() } {
                // SAFETY: a binding of the argument: a value's name, or a constant.
                if let Some(n) = unsafe { binding.name() } {
                    reads.push(n.to_string());
                }
            }
        }
        planned.push(Planned {
            operator,
            device,
            reads,
            makes,
        });
    }
    Ok(planned)
}

fn kind(d: &ProtocolObject<dyn MLComputeDeviceProtocol>) -> &'static str {
    if d.isKindOfClass(MLNeuralEngineComputeDevice::class()) {
        "ane"
    } else if d.isKindOfClass(MLGPUComputeDevice::class()) {
        "gpu"
    } else if d.isKindOfClass(MLCPUComputeDevice::class()) {
        "cpu"
    } else {
        "other"
    }
}

fn load(compiled: &Path, units: Units) -> Result<Retained<MLModel>, String> {
    // SAFETY: a file URL of a compiled model, and a configuration.
    unsafe {
        MLModel::modelWithContentsOfURL_configuration_error(
            &file_url(compiled),
            &configuration(units),
        )
    }
    .map_err(|e| format!("loading: {}", described(&e)))
}

fn shape_array(shape: &[usize]) -> Retained<NSArray<NSNumber>> {
    let v: Vec<Retained<NSNumber>> = shape.iter().map(|&d| NSNumber::new_usize(d)).collect();
    NSArray::from_retained_slice(&v)
}

/// An fp16 array of this shape, holding these bits in C order.
fn input_array(shape: &[usize], bits: &[u16]) -> Result<Retained<MLMultiArray>, String> {
    if bits.len() != shape.iter().product::<usize>() {
        return Err(format!("{} elements for a shape {shape:?}", bits.len()));
    }
    // SAFETY: a shape and the data type, fp16, Core ML takes.
    let array = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &shape_array(shape),
            MLMultiArrayDataType::Float16,
        )
    }
    .map_err(|e| described(&e))?;
    let written = RefCell::new(Err("not written".to_owned()));
    let block = block2::RcBlock::new(
        |bytes: NonNull<c_void>, size: NSInteger, strides: NonNull<NSArray<NSNumber>>| {
            // SAFETY: the array's strides, valid within the handler.
            let strides = strides_of(unsafe { strides.as_ref() });
            // SAFETY: the array's bytes, `size` of them, valid and ours
            // within the handler.
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    bytes.as_ptr().cast::<u8>(),
                    usize::try_from(size).unwrap_or(0),
                )
            };
            let mut result = Ok(());
            for (i, &b) in bits.iter().enumerate() {
                let at = 2 * offset(shape, &strides, i);
                match dst.get_mut(at..at + 2) {
                    Some(d) => d.copy_from_slice(&b.to_le_bytes()),
                    None => result = Err("an element past the array's bytes".to_owned()),
                }
            }
            *written.borrow_mut() = result;
        },
    );
    // SAFETY: a handler that writes only within the bytes it is given.
    unsafe { array.getMutableBytesWithHandler(&block) };
    drop(block);
    written.into_inner()?;
    Ok(array)
}

/// An output as the prediction left it: its elements' values in C order,
/// and their bits.
#[derive(Default)]
struct Got {
    values: Vec<f64>,
    bits: Vec<u64>,
}

fn read_array(array: &MLMultiArray) -> Result<(Vec<usize>, Got), String> {
    // SAFETY: the array's shape and strides, of a prediction's output.
    let (shape, strides) = unsafe { (array.shape(), array.strides()) };
    let (shape, strides) = (strides_of(&shape), strides_of(&strides));
    // SAFETY: as above.
    let ty = unsafe { array.dataType() };
    let size = match ty {
        MLMultiArrayDataType::Float16 => 2,
        MLMultiArrayDataType::Float32 | MLMultiArrayDataType::Int32 => 4,
        MLMultiArrayDataType::Float64 => 8,
        _ => return Err(format!("an output of data type {}", ty.0)),
    };
    let got = RefCell::new(Err("not read".to_owned()));
    let block = block2::RcBlock::new(|bytes: NonNull<c_void>, len: NSInteger| {
        // SAFETY: the array's bytes, `len` of them, valid within the handler.
        let all = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr().cast::<u8>(),
                usize::try_from(len).unwrap_or(0),
            )
        };
        let mut out = Got::default();
        let count: usize = shape.iter().product();
        for i in 0..count {
            let at = offset(&shape, &strides, i) * size;
            let Some(b) = all.get(at..at + size) else {
                *got.borrow_mut() = Err("an element past the array's bytes".to_owned());
                return;
            };
            let (value, bits) = match ty {
                MLMultiArrayDataType::Float16 => {
                    let h = u16::from_le_bytes([b[0], b[1]]);
                    (half_to_f64(h), u64::from(h))
                }
                MLMultiArrayDataType::Float32 => {
                    let x = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    (f64::from(x), u64::from(x.to_bits()))
                }
                MLMultiArrayDataType::Int32 => {
                    let x = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    (f64::from(x), u64::from(x.cast_unsigned()))
                }
                _ => {
                    let x = f64::from_le_bytes(b.try_into().expect("eight bytes"));
                    (x, x.to_bits())
                }
            };
            out.values.push(value);
            out.bits.push(bits);
        }
        *got.borrow_mut() = Ok(out);
    });
    // SAFETY: a handler that reads only within the bytes it is given.
    unsafe { array.getBytesWithHandler(&block) };
    drop(block);
    Ok((shape, got.into_inner()?))
}

fn dictionary<'a>(
    entries: impl Iterator<Item = (&'a str, Retained<AnyObject>)>,
) -> Retained<NSDictionary<NSString, AnyObject>> {
    let (keys, values): (Vec<Retained<NSString>>, Vec<Retained<AnyObject>>) =
        entries.map(|(k, v)| (NSString::from_str(k), v)).unzip();
    let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
    NSDictionary::from_retained_objects(&key_refs, &values)
}

/// One prediction: every output the case names.
fn predict(
    model: &MLModel,
    inputs: &[(String, Retained<MLMultiArray>)],
    outputs: &[Output],
) -> Result<BTreeMap<String, Got>, String> {
    let features = dictionary(inputs.iter().map(|(n, a)| {
        // SAFETY: a feature value of an array that outlives the prediction.
        let value = unsafe { MLFeatureValue::featureValueWithMultiArray(a) };
        (
            n.as_str(),
            Retained::into_super(Retained::into_super(value)),
        )
    }));
    // SAFETY: names to feature values, as the initializer takes.
    let provider = unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            &features,
        )
    }
    .map_err(|e| described(&e))?;
    let provider: Retained<ProtocolObject<dyn MLFeatureProvider>> =
        ProtocolObject::from_retained(provider);
    // SAFETY: a loaded model and its inputs.
    let out = unsafe { model.predictionFromFeatures_error(&provider) }
        .map_err(|e| format!("predicting: {}", described(&e)))?;
    let mut got = BTreeMap::new();
    for o in outputs {
        // SAFETY: an output the model declares, read after the prediction.
        let value = unsafe { out.featureValueForName(&NSString::from_str(&o.name)) }
            .ok_or_else(|| format!("no output {}", o.name))?;
        // SAFETY: as above.
        let array =
            unsafe { value.multiArrayValue() }.ok_or_else(|| format!("{} is no array", o.name))?;
        let (shape, g) = read_array(&array)?;
        if shape != o.shape {
            return Err(format!("{}: shape {shape:?}, not {:?}", o.name, o.shape));
        }
        got.insert(o.name.clone(), g);
    }
    Ok(got)
}

/// What a run on one set of compute units found: its lines, and whether
/// it failed.
/// Where the case's ops run: its line of them, and what breaks the rule.
fn check_placement(case: &Case, compiled: &Path, units: Units) -> (String, Vec<String>) {
    match placement(compiled, units) {
        Ok(planned) => {
            let mut by_device: BTreeMap<&str, usize> = BTreeMap::new();
            for op in &planned {
                *by_device.entry(op.device).or_default() += 1;
            }
            let counts: Vec<String> = by_device.iter().map(|(d, n)| format!("{d} {n}")).collect();
            let line = format!(
                "{} [{}]: ops {}",
                case.name,
                units.name(),
                counts.join(", ")
            );
            let wrong = if units == Units::NeuralEngine {
                misplaced(&planned)
            } else {
                Vec::new()
            };
            (line, wrong)
        }
        Err(e) => (format!("{} [{}]", case.name, units.name()), vec![e]),
    }
}

/// An output held to its rule: its line, and what breaks it.
fn check_output(
    case: &Case,
    o: &Output,
    first: &BTreeMap<String, Got>,
    units: Units,
) -> (String, Vec<String>) {
    let got = &first[&o.name];
    let eval = read_f64(&case.dir.join("eval").join(format!("{}.f64", o.name)));
    let mut wrong = Vec::new();
    match &o.kind {
        Kind::Float(hold) => {
            let interp = read_f64(&case.dir.join("interp").join(format!("{}.f64", o.name)));
            let model = read_f64(&case.dir.join("model").join(format!("{}.f64", o.name)));
            let (tau, floor) = units.bound(*hold);
            let mut said = Vec::new();
            if *hold == Hold::Model {
                if units != Units::NeuralEngine {
                    wrong.push(format!(
                        "{}: held to the Neural Engine's model on the CPU",
                        o.name
                    ));
                }
                let (count, first) = model_errors(&got.values, &model);
                said.push(format!("{count} of {} from the model", got.values.len()));
                if let Some(i) = first {
                    wrong.push(format!(
                        "{}: {count} elements from the model, the first {i}: {} against {}",
                        o.name, got.values[i], model[i]
                    ));
                }
            }
            for (reference, want) in [("eval", &eval), ("interp", &interp)] {
                let (worst, row) = float_error(&got.values, want, o.rows(), floor);
                said.push(format!("{reference} {:.3}τ (row {row})", worst / tau));
                if *hold != Hold::Model && (worst > tau || worst.is_nan()) {
                    wrong.push(format!(
                        "{}: row {row} {:.3}τ from {reference}",
                        o.name,
                        worst / tau
                    ));
                }
            }
            // The interpreter computing as the Neural Engine does, as far as
            // it was observed, against the evaluator, held to nothing.
            let (worst, row) = float_error(&model, &eval, o.rows(), floor);
            said.push(format!("the model {:.3}τ (row {row})", worst / tau));
            let how = match hold {
                Hold::Plain => String::new(),
                Hold::Softcap => " (through a softcap)".to_owned(),
                Hold::Model => " (held to the model)".to_owned(),
            };
            (
                format!("  {}{how}, τ {tau}: {}", o.name, said.join(", ")),
                wrong,
            )
        }
        Kind::Token(rule) => {
            let logits = match rule {
                Token::Logits(n) => first.get(n).map(|l| (l.values.as_slice(), n)),
                Token::Exact => None,
            };
            let reference_logits =
                logits.map(|(_, n)| read_f64(&case.dir.join("eval").join(format!("{n}.f64"))));
            // The logits' own bound.
            let hold = match rule {
                Token::Logits(n) => case.outputs.iter().find_map(|l| match l.kind {
                    Kind::Float(h) if &l.name == n => Some(h),
                    _ => None,
                }),
                Token::Exact => None,
            };
            let (tau, _) = units.bound(hold.unwrap_or(Hold::Plain));
            let errors = token_errors(
                rule,
                &got.values,
                &eval,
                logits
                    .zip(reference_logits.as_deref())
                    .map(|((p, _), r)| (p, r)),
                tau,
            );
            let line = format!(
                "  {}: {} rows, {} wrong",
                o.name,
                got.values.len(),
                errors.len()
            );
            wrong.extend(errors.into_iter().map(|e| format!("{}: {e}", o.name)));
            (line, wrong)
        }
    }
}

/// With `ANE_CONFORMANCE_KEEP` set, every output as predicted, for a look:
/// `<dir>/<case>.<units>/<name>.f64`.
fn keep(case: &Case, units: Units, got: &BTreeMap<String, Got>) {
    if let Ok(keep) = env::var("ANE_CONFORMANCE_KEEP") {
        let dir = Path::new(&keep).join(format!("{}.{}", case.name, units.name()));
        fs::create_dir_all(&dir).ok();
        for (name, g) in got {
            let bytes: Vec<u8> = g.values.iter().flat_map(|v| v.to_le_bytes()).collect();
            fs::write(dir.join(format!("{name}.f64")), bytes).ok();
        }
    }
}

/// What a run on one set of compute units found: its lines, and whether
/// it failed.
fn run_units(case: &Case, compiled: &Path, units: Units) -> (Vec<String>, bool) {
    let mut wrong = Vec::new();
    let (line, misplaced) = check_placement(case, compiled, units);
    let mut lines = vec![line];
    wrong.extend(misplaced);
    let predicted = load(compiled, units).and_then(|model| {
        let inputs = case
            .inputs
            .iter()
            .map(|(n, shape)| {
                input_array(shape, &read_f16(&case.dir.join(format!("{n}.f16"))))
                    .map(|a| (n.clone(), a))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let first = predict(&model, &inputs, &case.outputs)?;
        let second = predict(&model, &inputs, &case.outputs)?;
        Ok((first, second))
    });
    match predicted {
        Ok((first, second)) => {
            keep(case, units, &first);
            for o in &case.outputs {
                if first[&o.name].bits != second[&o.name].bits {
                    wrong.push(format!("{}: two predictions differ", o.name));
                }
                let (line, w) = check_output(case, o, &first, units);
                lines.push(line);
                wrong.extend(w);
            }
        }
        Err(e) => wrong.push(e),
    }
    let failed = !wrong.is_empty();
    lines.extend(wrong.into_iter().map(|w| format!("  FAILED: {w}")));
    (lines, failed)
}

fn run_case(case: &Case) -> (Vec<String>, bool) {
    let compiled = match compile(case) {
        Ok(c) => c,
        Err(e) => return (vec![format!("{}: FAILED: {e}", case.name)], true),
    };
    let mut lines = Vec::new();
    let mut failed = false;
    for &units in &case.units {
        let (l, f) = autoreleasepool(|_| run_units(case, &compiled, units));
        lines.extend(l);
        failed |= f;
    }
    fs::remove_dir_all(&compiled).ok();
    (lines, failed)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let [corpus] = args.as_slice() else {
        fail("usage: furykernel-ane-conformance CORPUS")
    };
    println!(
        "macOS: {}",
        NSProcessInfo::processInfo().operatingSystemVersionString()
    );
    let mut dirs: Vec<PathBuf> = fs::read_dir(corpus)
        .unwrap_or_else(|e| fail(&format!("{corpus}: {e}")))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("outputs.txt").is_file())
        .collect();
    dirs.sort();
    let mut failed = 0;
    for dir in &dirs {
        let (lines, f) = run_case(&parse_case(dir));
        for l in lines {
            println!("{l}");
        }
        failed += usize::from(f);
    }
    println!("{} cases, {failed} failed", dirs.len());
    if failed > 0 || dirs.is_empty() {
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves() {
        assert_eq!(half_to_f64(0x3c00).to_bits(), 1.0_f64.to_bits());
        assert_eq!(half_to_f64(0xc000).to_bits(), (-2.0_f64).to_bits());
        assert_eq!(half_to_f64(0x0001).to_bits(), 2f64.powi(-24).to_bits());
        assert_eq!(half_to_f64(0x0400).to_bits(), 2f64.powi(-14).to_bits());
        assert_eq!(half_to_f64(0x7bff).to_bits(), 65_504.0_f64.to_bits());
        assert_eq!(half_to_f64(0x8000).to_bits(), (-0.0_f64).to_bits());
        assert!(half_to_f64(0x7c00).is_infinite());
        assert!(half_to_f64(0x7e00).is_nan());
    }

    #[test]
    fn offsets_follow_strides() {
        // A [2, 3] array whose rows are padded to four elements.
        let shape = [2, 3];
        let strides = [4, 1];
        let at: Vec<usize> = (0..6).map(|i| offset(&shape, &strides, i)).collect();
        assert_eq!(at, [0, 1, 2, 4, 5, 6]);
    }

    #[test]
    fn rows_are_held_to_their_own_magnitude() {
        // Two rows (the last axis) of three elements, element k of row r at
        // 2k + r: row 0 of largest magnitude 4, row 1 of 4000.
        let want = [1.0, 1000.0, 2.0, 2000.0, -4.0, 4000.0];
        // Row 0 off by 2τ, half the bound of its largest; row 1 exact.
        let floor = ABSOLUTE / TAU;
        let mut got = want;
        got[0] += 2.0 * TAU;
        let (worst, row) = float_error(&got, &want, 2, floor);
        assert!((worst - TAU / 2.0).abs() < 1e-12);
        assert_eq!(row, 0);
        // 20τ is too much for row 0, though row 1's magnitude would allow it.
        got[0] = want[0] + 20.0 * TAU;
        assert!(float_error(&got, &want, 2, floor).0 > TAU);
    }

    #[test]
    fn bounds_by_units_and_kind() {
        // The Neural Engine: 2^-7, a softcap's output 2^-6, either 2^-9 at
        // the least; the CPU 2^-5 of a row's largest or of 2^-3.
        assert_eq!(Units::NeuralEngine.bound(Hold::Plain), (1.0 / 128.0, 0.25));
        assert_eq!(
            Units::NeuralEngine.bound(Hold::Softcap),
            (1.0 / 64.0, 0.125)
        );
        assert_eq!(Units::NeuralEngine.bound(Hold::Model), (1.0 / 128.0, 0.25));
        assert_eq!(Units::Cpu.bound(Hold::Plain), (1.0 / 32.0, 0.125));
        assert_eq!(Units::Cpu.bound(Hold::Softcap), (1.0 / 32.0, 0.125));
        for hold in [Hold::Plain, Hold::Softcap] {
            let (tau, floor) = Units::NeuralEngine.bound(hold);
            assert_eq!(tau * floor, 1.0 / 512.0);
        }
    }

    #[test]
    fn outputs_held_to_the_model() {
        let model = [1.0, -0.0, f64::INFINITY, f64::NAN, 2f64.powi(-16)];
        // Its own values, whatever their signs of zero, NaNs alike.
        assert_eq!(
            model_errors(&[1.0, 0.0, f64::INFINITY, f64::NAN, 2f64.powi(-16)], &model),
            (0, None)
        );
        // A grid step off, or finite where it overflows.
        assert_eq!(
            model_errors(&[1.0, 0.0, 65_504.0, f64::NAN, 0.0], &model),
            (2, Some(2))
        );
        // Short of it.
        assert_eq!(model_errors(&[1.0], &model).0, 4);
    }

    #[test]
    fn small_rows_are_held_to_the_floor_and_nonfinite_values_fail() {
        let want = [0.0, 1.0, 0.0, 1.0];
        let mut got = want;
        // A row of zeros, held to ABSOLUTE.
        let floor = ABSOLUTE / TAU;
        got[2] = ABSOLUTE / 2.0;
        let (worst, row) = float_error(&got, &want, 2, floor);
        assert!((worst - TAU / 2.0).abs() < 1e-12 && row == 0);
        let mut got = want;
        got[3] = f64::NAN;
        let (worst, row) = float_error(&got, &want, 2, floor);
        assert!(worst.is_infinite() && row == 1);
        let mut got = want;
        got[0] = -0.0;
        assert_eq!(float_error(&got, &want, 2, floor).0, 0.0);
    }

    #[test]
    fn tokens_of_logits() {
        // Two rows of four logits, element k of row r at 2k + r: row 0's
        // reference logits 1, 3, 3, 0, and row 1's 5, 5, 1, 0.
        let reference = [1.0, 5.0, 3.0, 5.0, 3.0, 1.0, 0.0, 0.0];
        let rule = Token::Logits("logits".to_owned());
        // Either of row 0's tying 3s is right, but only as the first largest
        // of the predicted logits, here element 1; row 1's first 5.
        let predicted = [1.0, 5.0, 3.0, 5.0, 3.0 - 1.0 / 256.0, 1.0, 0.0, 0.0];
        assert_eq!(
            token_errors(
                &rule,
                &[1.0, 0.0],
                &[2.0, 0.0],
                Some((&predicted, &reference)),
                TAU
            ),
            Vec::<String>::new()
        );
        // Not the first largest of the predicted logits.
        assert_eq!(
            token_errors(
                &rule,
                &[2.0, 0.0],
                &[2.0, 0.0],
                Some((&predicted, &reference)),
                TAU
            )
            .len(),
            1
        );
        // The first largest of its predicted logits, but its reference
        // logit, 2.5, too far below the largest, 3.
        let far = [1.0, 5.0, 3.0, 5.0, 2.5, 1.0, 0.0, 0.0];
        let predicted_far = [1.0, 5.0, 2.0, 5.0, 3.0, 1.0, 0.0, 0.0];
        assert_eq!(
            token_errors(
                &rule,
                &[2.0, 0.0],
                &[1.0, 0.0],
                Some((&predicted_far, &far)),
                TAU
            )
            .len(),
            1
        );
        // Within twice a softcap's bound of 3, 2^-6, but not twice 2^-7.
        let near = [1.0, 5.0, 3.0, 5.0, 2.92, 1.0, 0.0, 0.0];
        assert_eq!(
            token_errors(
                &rule,
                &[2.0, 0.0],
                &[1.0, 0.0],
                Some((&predicted_far, &near)),
                TAU
            )
            .len(),
            1
        );
        assert_eq!(
            token_errors(
                &rule,
                &[2.0, 0.0],
                &[1.0, 0.0],
                Some((&predicted_far, &near)),
                TAU_SOFTCAP
            ),
            Vec::<String>::new()
        );
    }

    #[test]
    fn exact_tokens_and_bad_indices() {
        assert_eq!(
            token_errors(&Token::Exact, &[3.0, 16_383.0], &[3.0, 16_383.0], None, TAU),
            Vec::<String>::new()
        );
        assert_eq!(
            token_errors(&Token::Exact, &[3.0, 16_384.0], &[3.0, 16_383.0], None, TAU).len(),
            1
        );
        assert_eq!(
            token_errors(&Token::Exact, &[-1.0, 0.5], &[0.0, 0.0], None, TAU).len(),
            2
        );
    }

    #[test]
    fn only_the_tail_leaves_the_neural_engine() {
        let op = |operator: &str, device, reads: &[&str], makes: &str| Planned {
            operator: operator.to_owned(),
            device,
            reads: reads.iter().map(|r| (*r).to_owned()).collect(),
            makes: vec![makes.to_owned()],
        };
        // The chunks joined and argmaxed on the CPU, at the end: right.
        let tail = [
            op("conv", "ane", &["x"], "a"),
            op("conv", "ane", &["x"], "b"),
            op("concat", "cpu", &["a", "b"], "logits"),
            op("reduce_argmax", "cpu", &["logits"], "token"),
        ];
        assert_eq!(misplaced(&tail), Vec::<String>::new());
        // A concat whose value the Neural Engine reads again: wrong, as is
        // any other op off it.
        let middle = [
            op("concat", "cpu", &["x", "y"], "c"),
            op("layer_norm", "ane", &["c"], "n"),
            op("mul", "cpu", &["n"], "m"),
            op("cast", "fused", &["m"], "f"),
        ];
        assert_eq!(misplaced(&middle).len(), 2);
    }

    #[test]
    fn output_lines() {
        assert_eq!(
            parse_output("soft float 1,2816,1,28"),
            Output {
                name: "soft".to_owned(),
                kind: Kind::Float(Hold::Plain),
                shape: vec![1, 2816, 1, 28]
            }
        );
        assert_eq!(
            parse_output("logits float 1,20000,1,4 softcap").kind,
            Kind::Float(Hold::Softcap)
        );
        assert_eq!(
            parse_output("y float 1,1024,1,256 model").kind,
            Kind::Float(Hold::Model)
        );
        assert_eq!(
            parse_output("next_token token 1,1,1,4 logits"),
            Output {
                name: "next_token".to_owned(),
                kind: Kind::Token(Token::Logits("logits".to_owned())),
                shape: vec![1, 1, 1, 4]
            }
        );
        assert_eq!(
            parse_output("token token 1,1,1,4 exact").kind,
            Kind::Token(Token::Exact)
        );
    }
}
