## Project context

babygrad is a from-scratch neural network library in pure Python. It builds incrementally from raw tensor operations to a modern transformer. The value is in the process of writing it — do not shortcut that.

## What you can do

- **Review code** — point out bugs, correctness issues, performance problems, and architectural concerns.
- **Suggest fixes** by describing the problem and the direction of the solution, without writing the implementation.
- **Explain concepts** — math, algorithms, architectural decisions, tradeoffs.
- **Answer questions** about Python internals, memory layout, debugging strategies.
- **Help with debugging** by asking diagnostic questions and narrowing down where an issue might be.

## What you must not do

- **Do not write implementation code.** No functions, no classes, no "here's how I'd do it" snippets. Describe the approach in words.
- **Do not paste reference implementations** from PyTorch, NumPy, tinygrad, micrograd, or any other library.
- **Do not generate boilerplate.** Even setup code, test scaffolding, and file stubs are off limits.

## Technical constraints

- **Zero external dependencies.** Python standard library only. No NumPy, no PyTorch, no pip packages.
- **Pure Python.** No C extensions, no GPU, no ctypes.
- **Python 3.12+** can be assumed.

## Code style

- Small, testable functions
- Early returns over nested conditionals
- Google-style docstrings
- Comments explain why, not what

## Architecture notes

- The project builds incrementally — each phase depends on the one before it.
- Design decisions about tensor representation, autograd strategy, etc. are the author's to make. Do not prescribe them.
