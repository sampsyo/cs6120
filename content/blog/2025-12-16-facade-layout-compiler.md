+++
title = "Facade Layout Compiler (# 618)"
[extra]
bio = """
  I-Ting Tsai is a CS PhD student at Cornell University.
"""
[[extra.authors]]
name = "I-Ting Tsai"
+++

# Experience Report: Facade Layout Compiler

[Source Code](https://github.com/itingtsai/Facade_Layout_Compiler)

## What was the goal?

The goal of this project was to design and implement a Facade Layout Compiler—a domain-specific compiler that transforms textual descriptions of building facades into verified, renderable outputs. The input language describes a facade as a symbolic grid of architectural elements (windows, doors, chimneys, spacing, and empty cells), and the compiler produces:

**Verified layouts:** The compiler performs static semantic analysis to enforce architectural constraints such as door placement rules, chimney connectivity, and optional window spacing requirements.

**Renderable outputs:** For valid programs, the compiler generates both SVG visualizations and structured JSON representations suitable for downstream processing or generative design workflows.

This project directly mirrors a traditional compiler pipeline—lexical analysis, parsing, semantic analysis, and code generation—but operates in a visually interpretable domain. The choice of architectural facades as the target domain was intentional: it provides immediate visual feedback for correctness, making it easier to reason about both the compiler's behavior and the quality of its output.

## Alignment with Proposal

My original proposal outlined three key deliverables:

1. **Custom DSL for facade specification:** Implemented with support for row declarations, repeat expressions, compact syntax, comments, and rule declarations
2. **Static rule enforcement:** Distinct semantic checks with error/warning classification
3. **Empirical validation through testing:** pass a comprehensive test suite

The scope evolved slightly during implementation. More details are described in the sections below.

---

## What did you do?

The complete implementation is available in [facade_compiler.py](https://github.com/itingtsai/Facade_Layout_Compiler/blob/main/facade_compiler.py)

### DSL Design

The facade DSL is designed as a minimal, domain-specific language for describing building facades as discrete 2D grids. Each facade is expressed as a set of numbered row declarations, where each row contains a sequence of symbolic elements:

- `E` = empty space
- `W` = window
- `S` = spacing
- `D` = door
- `C` = chimney

**Example program:**

```
row 1: E E E E E E E E
row 2: E E E E E E E E
row 3: E W S S S S W E
row 4: E W S S S S W E
row 5: E E E D D E E E
```

This grid-based representation provides a clear correspondence between source-level specification and spatial layout. The choice of single-character symbols was deliberate: it enables the compact syntax feature (described below) while remaining visually scannable.

### Expressiveness and Usability Features

To balance expressiveness with simplicity, the DSL supports several usability-oriented features:

1. **Compact syntax:** Symbols may be written without spaces (e.g., `EWCWE`), allowing concise row specifications. The lexer recognizes sequences of valid symbols and expands them into individual cells during tokenization.

2. **Repeat expressions:** Elements can be repeated using `SYMBOL*COUNT` syntax (e.g., `W*5` expands to five window cells). This significantly reduces verbosity for large facades and was particularly useful for stress testing.

3. **Zero-count repeats:** The expression `E*0` produces zero cells, effectively allowing conditional exclusion. This edge case required careful handling—the parser simply doesn't add any cells to the AST rather than treating it as an error.

4. **Flexible whitespace and comments:** Arbitrary whitespace (spaces, tabs, newlines) is permitted between tokens. Line comments starting with `#` are ignored by the lexer. This allows users to format facade programs for clarity.

5. **Case insensitivity:** Both `E` and `e` are valid (normalized to uppercase internally).

6. **Rule declarations:** Optional rule declarations (e.g., `rule min_window_spacing: 2`) parameterize semantic checks. Rules are stored in the IR and consulted during analysis.

7. **Auto-fill behavior:** Rows may have different lengths; the compiler automatically pads shorter rows with empty cells to create a rectangular grid. Auto-filled cells are tracked in metadata so users can distinguish compiler-inferred content from explicit specification.

### Design Rationale

The DSL intentionally avoids geometric coordinates, numeric dimensions, or continuous values. By focusing on symbolic structure, the language:

- Simplifies parsing and semantic analysis
- Enables clear error reporting at the language level
- Allows the same IR to be reused for multiple back-end targets (SVG, JSON, or potentially image generation prompts)

The grammar is regular and unambiguous, making it amenable to simple hand-written parsing without requiring a parser generator.

---

## Front-End: Lexer and Parser

### Lexer Implementation

I implemented a hand-written lexer that performs character-level scanning. The lexer maintains source location tracking (line and column) for error reporting.

**Token types recognized:**

- Keywords: `row`, `rule`
- Symbols: `E`, `W`, `S`, `D`, `C`
- Integers (for row numbers and repeat counts)
- Punctuation: `:` (colon), `*` (multiply for repeats)
- Identifiers (for rule names and symbol sequences)

**Key design decisions:**

- **Invalid character detection:** The lexer explicitly rejects characters not in the DSL's alphabet (punctuation, Unicode, emoji). This catches input errors immediately rather than propagating malformed tokens to the parser.

- **Negative number handling:** Rather than tokenizing `-` as a separate operator, the lexer detects the pattern `-[digit]` and reports "Negative numbers not allowed" directly. This provides clearer error messages than a generic parse error.

- **Symbol sequence recognition:** When the lexer encounters a multi-character word like `EWCWE`, it checks if all characters are valid symbols. If so, it emits a single `IDENTIFIER` token that the parser later expands. This enables compact syntax without complicating the grammar.

- **Error handling:** Lexical errors are reported but do not prevent continued scanning. This allows the compiler to report multiple errors in a single pass, improving the user experience during iterative development.

### Parser Implementation

The parser is a hand-written top-down predictive parser that consumes the token stream and constructs an AST consisting of:

- **Program:** The root node containing rules and rows
- **RuleDecl:** A rule declaration with name and value(s)
- **RowDecl:** A row with its number and list of cells
- **Cell:** A single symbol at a source location
- **RepeatExpr:** A symbol with a repeat count

**Grammar (EBNF):**

```
program     ::= rule_decl* row_decl+
rule_decl   ::= 'rule' IDENTIFIER ':' rule_value+
row_decl    ::= 'row' INTEGER ':' cell+
cell        ::= SYMBOL | repeat_expr | symbol_sequence
repeat_expr ::= SYMBOL '*' INTEGER
```

**Note on Grammar Classification:** This grammar is regular, and the structure is purely sequential without nesting. The language could equivalently be expressed as a regular expression:

```
(rule IDENTIFIER : value+)* (row INTEGER : (SYMBOL | SYMBOL * INTEGER)+)+
```

**Note on Parser Classification:** The parser follows a top-down, predictive parsing strategy with one function per major grammar production (`parse()`, `parse_rule()`, `parse_row()`, `parse_cells()`). It is not strictly a recursive-descent parser because no function calls itself—the grammar's repetition operators (`*` and `+`) are implemented with iterative `while` loops rather than recursive calls. This iterative approach is natural for a regular grammar where recursion is unnecessary. The parser uses single-token lookahead via `check()` to predict which production to apply, making it a predictive parser.

**Key design decisions:**

- **Error recovery:** When a syntax error is detected (e.g., missing colon), the parser calls `sync_to_row()` to skip tokens until the next `row` keyword. This prevents cascading errors and allows multiple independent errors to be reported.

- **Repeat expression handling:** The `*0` case is handled by simply not adding any cell to the AST—the absence of cells is the correct representation of "repeat zero times."

- **Duplicate row handling:** When the same row number appears twice, the parser accepts both, only the latest duplicated row would show, and the semantic analyzer emits a warning. This is more user-friendly than treating duplicates as errors.

- **Strict validation:** Row numbers must be positive integers. Syntax errors like `row -1:`, `row 1.5:`, or `row :` are caught and reported with helpful hints.

---

## Middle-End: Semantic Analysis and Normalization

The semantic analyzer transforms the parsed AST into a normalized grid representation and enforces architectural constraints.

### Grid Construction and Normalization

**Step 1: Build initial grid**

The analyzer iterates over row declarations (sorted by row number) and populates a 2D array. If row numbers have gaps (e.g., rows 1, 3, 5 are specified), intermediate rows are initialized as empty lists.

**Step 2: Normalize grid dimensions**

After construction, the grid may be jagged (rows of different lengths). The normalizer:

- Computes the maximum row width
- Pads shorter rows with `E` (empty) cells
- Records which cells were auto-filled in metadata

This guarantees a rectangular grid for all downstream processing.

### Architectural Rule Checks

#### Door Rule

Doors have strict placement requirements that mirror real-world construction:

- All door cells in a column must form a contiguous vertical segment
- The segment must extend to the ground floor (bottom row)

The implementation:

1. Groups door cells by column
2. For each column, verifies the bottom row is included
3. Checks for gaps by ensuring consecutive row indices differ by exactly 1

Violations produce errors because a floating or disconnected door is structurally invalid.

#### Chimney Rule

Chimneys present a more complex connectivity problem:

- Chimney cells may form arbitrary connected components (including diagonal patterns via 4-connectivity)
- Each component must connect to the roof (top row)

The implementation uses flood-fill to identify connected components. For each component, the analyzer checks if any cell has row index 0 (top row). Components that don't reach the roof produce errors.

#### Window Spacing Rule

If the rule `min_window_spacing` is declared, the analyzer checks horizontal distances between adjacent windows in each row. Unlike door and chimney rules, spacing violations produce warnings only rather than errors, because tight window spacing may be intentional in some architectural styles.

#### Symmetry Detection

The analyzer compares each row with its horizontal mirror. Asymmetric rows produce warnings to inform the designer of potential stylistic inconsistency. This is informational rather than prescriptive.

### Error Classification

I deliberately distinguished between errors (fatal violations) and warnings (stylistic concerns):

- **Error:** Structurally invalid facade. Compilation halts; no output generated.
- **Warning:** Stylistic issue or optional constraint violation. Compilation continues; warning recorded in output.
- **Info:** Informational message (e.g., auto-fill notification). No effect on compilation.

This classification prevents the compiler from being overly restrictive while still catching genuine architectural errors.

---

## Back-End: Code Generation

For valid programs, the compiler generates two output formats:

### JSON Output

The JSON output includes:

- Grid metadata (dimensions, rules, auto-filled cells)
- The normalized grid as a 2D array of symbols
- Validation results (is_valid, errors, warnings)

This format is designed for programmatic consumption. The structured data can be used by downstream tools for further processing, constraint solving, or generative design.

### SVG Output

The SVG generator produces a visual representation where each cell is rendered as a colored rectangle with a label. Cells are grouped by type in the SVG DOM, making it easy to style or manipulate specific element types.

### PNG Rasterization

For environments that don't support SVG, the compiler can produce PNG output via CairoSVG.

---

## What were the hardest parts to get right?

### 1. Separating Syntax Errors from Semantic Errors

The distinction between lexical, syntactic, and semantic errors was initially blurry. For example:

- Is `@` an invalid character (lexical) or an invalid symbol (semantic)?
- Is `row 1: X E E` a syntax error or a semantic error?

I resolved this by establishing boundaries:

- **Lexical errors:** Characters not in the DSL's alphabet (punctuation, Unicode, emoji)
- **Syntax errors:** Malformed constructs (missing colons, invalid repeat expressions, missing row numbers)
- **Semantic errors:** Valid syntax with invalid meaning (floating doors, disconnected chimneys, invalid symbol letters like `X`)

The key insight was that `X` is a valid identifier syntactically but not a valid symbol semantically. The parser accepts it and emits a semantic error during cell processing.

### 2. Chimney Connectivity Analysis

Initially, I tried a simple positional check (e.g., chimney cells must be on the top row). This failed for common patterns like diagonal chimneys:

```
row 1: C E E
row 2: C C E
row 3: E C C
row 4: E E C
```

This chimney is valid. It connects to the roof through a diagonal path. The flood-fill approach correctly handles this by treating chimneys as connected components rather than isolated cells.

A subtle bug emerged with multiple chimney components: the algorithm must check each component independently, not just verify that some chimney reaches the roof.

### 3. Zero-Count Repeat Expressions

The expression `E*0` presented an interesting design decision. Options:

1. Treat as syntax error (reject)
2. Treat as single cell (emit one `E`)
3. Treat as zero cells (emit nothing)

I chose option 3 because it's the most mathematically consistent interpretation and enables useful patterns like conditional exclusion. The implementation simply doesn't add any cell to the AST when count is 0.

### 4. Normalization Without Hiding Bugs

Auto-filling rows and padding widths improved usability but introduced a risk: it could mask user errors. If a user forgets a cell, the compiler silently adds an empty cell, which potentially hiding bugs.

I addressed this by:

- Explicitly tracking auto-filled cells in metadata
- Emitting INFO messages about auto-fill actions
- Including auto-fill information in JSON output

This makes compiler intervention visible without treating it as an error.

### 5. Error Recovery and Multiple Error Reporting

Fixing one error often revealed another, requiring multiple compile-fix cycles.

I implemented error recovery in both the lexer and parser:

- **Lexer:** Reports invalid characters but continues scanning
- **Parser:** Uses `sync_to_row()` to skip to the next row declaration after errors

This required careful state management to ensure the compiler doesn't report spurious follow-on errors while still catching genuine independent errors.

### 6. Comprehensive Test Coverage

The most time-consuming aspect was ensuring comprehensive test coverage. My initial tests covered obvious paths but missed edge cases like:

- `E*0` in various positions
- Mixed case symbols
- Comments in unusual positions
- Unicode in comments (valid) vs. Unicode in code (invalid)
- Multiple errors in the same program

I used an LLM as a test ideation tool: after implementing a feature, I asked it to propose adversarial inputs and edge cases. I then checked and determined the expected behavior and added these cases to my test suite.

---

## Were you successful?

### Quantitative Evaluation

The testing results printout is available in [test.txt](https://github.com/itingtsai/Facade_Layout_Compiler/blob/main/test/test.txt)

Comprehensive testing is hard. There is no existing benchmark. After implementing and validating each new rule, I use an LLM for test ideation (e.g., "Give me edge cases to test user inputs containing characters outside the defined DSL elements.") and then manually determine and verify the expected behavior. This approach helped broaden test coverage to corner cases that I wouldn't have considered. The categorized test suite made it easy to verify that each feature worked in isolation and in combination.

### Test Suite Composition

The test suite comprises 146 facade programs organized into 22 categories:

| Category | Tests | Description |
|----------|-------|-------------|
| Basic | 5 | Simple valid inputs |
| Syntax | 11 | Syntax variations (compact, repeat, case) |
| Comments | 4 | Comment handling |
| AutoFill | 5 | Auto-fill behavior |
| RowNum | 5 | Row number handling |
| Rules | 4 | Rule declarations |
| Door | 8 | Door structural validation |
| Chimney | 8 | Chimney connectivity validation |
| WinSpace | 4 | Window spacing rule |
| BadChar | 18 | Invalid character detection |
| BadSymbol | 6 | Invalid symbol detection |
| SyntaxErr | 18 | Syntax error handling |
| Edge | 10 | Edge cases |
| Complex | 4 | Complex valid facades |
| Boundary | 4 | Boundary conditions |
| WS | 5 | Whitespace edge cases |
| BadSymbol2 | 6 | Additional invalid symbol tests |
| Stress | 5 | Stress tests (large grids, many rows) |
| Recovery | 3 | Error recovery |
| Combo | 4 | Feature combinations |
| ZeroRepeat | 9 | Zero-count repeat expressions |

All 146 tests pass, demonstrating:

- **Functional correctness:** The compiler accepts valid programs and rejects invalid ones
- **Rule enforcement precision:** Architectural constraints are enforced exactly as specified
- **Robustness:** The compiler handles syntactic variations, malformed input, and edge cases gracefully

### Stress Test Results

The compiler handles extreme inputs without issues:

- 100-column rows
- 100-row facades
- `E*9999` (large repeat)
- `E*500` repeated 10 times

### Qualitative Validation

I output 5 facade examples (json, svg, png) together with image gen visualizations which are available in the [example folder](https://github.com/itingtsai/Facade_Layout_Compiler/tree/main/example).

Beyond automated testing, I validated output quality through visual inspection:

- **SVG correctness:** Generated SVGs accurately represent the input grid with correct colors and positioning
- **Error message quality:** Error messages include source locations and actionable hints
- **Warning appropriateness:** Symmetry and spacing warnings provide useful feedback without blocking compilation

### Additional Experiment

I conducted an experiment using an image generation model (Nano Banana) to produce photorealistic facade images from compiled layouts. The workflow:

1. Compile facade DSL → JSON + SVG (+ PNG)
2. Construct prompt from grid structure and style description
3. Generate image using multimodal model

**Text prompt format:** *Generate a [-style] facade of a [low-rise/mid-rise/high-rise] building based on this image, where W = window, D = door, S = spacing, E = empty, and C = chimney.*

The results were correct most of the time. The model understood the general layout but sometimes struggled with precise spatial positioning. This suggests the compiled output could serve as input for generative design, though current models may need additional guidance.

---

## Conclusion

This project shows that the classical compiler pipeline—lexing, parsing, semantic analysis, and code generation—maps cleanly onto domain-specific languages, even in nontraditional domains such as architectural layout. Treating facades as programs rather than data made it possible to enforce structural invariants that would be difficult to express or validate in other formats.

Building the compiler highlighted several practical lessons: DSL design requires careful trade-offs between expressiveness and complexity; high-quality error messages are essential for usability; and comprehensive testing is challenging in the absence of established benchmarks, requiring systematic, feature-driven test design. The phased architecture proved especially valuable for debugging and extension, as clear phase boundaries made errors easy to localize and new features straightforward to add.

Overall, this experience reinforced the power of compiler techniques as a general problem-solving framework. The resulting facade compiler provides a solid foundation for future extensions, such as richer architectural constraints, integration with generative design systems, or interactive feedback during editing.

It was a great semester, even though I was personally juggling a lot. Coming from a very different background, I learned a lot from this course and developed a better understanding of what a compiler is and how it works. Thanks to everyone for the great online and in-class discussions, and especially to Adrian and Kei for the guidance and help throughout the semester.