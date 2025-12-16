+++
title = "ChocoPy2Bril: Translating ChocoPy to Bril"
[extra]
latex = false
bio = """
  Jackie, Maggie and Serena are all students in Advanced Compilers!
"""
[[extra.authors]]
name = "Jacqueline Wen"
[[extra.authors]]
name = "Maggie Gao"
[[extra.authors]]
name = "Serena Zhang"
+++

### What was the goal?
The primary goal is to create a compiler frontend, named "ChocoPy2Bril," that translates a core subset of the ChocoPy language directly into the Bril representation. This project will focus on the AST traversal and code generation phase by leveraging an existing ChocoPy parser to handle lexical analysis and syntax analysis.

Specifically, we supported the following functionalities of ChocoPy:
- Primitive Types: Integers and Booleans (int, bool).
- Basic Operations: Arithmetic (+, -, *, /), Relational (==, !=, <, >, <=, >=), and Logical (and, or, not).
- Variables: Local variable declaration and assignment.
- Control Flow: Conditional statements (if/else) and simple iteration (while loops).
- Functions: Function definition, argument passing, and return statements.

Note: we were originally also planning on supporting modulo, but due to Bril not having a modulo operator, we were forced to leave it out.

### What did you do?
For the simplicity of this project, we decided to focus on a subset of ChocoPy that includes primitive types, basic operations, variable declarations and assignments, control flow statements, and function definitions. We decided on these primatives as we felt that we could cover a reasonable amount of simple program functionalities alone with these primativs. 

Without too much difficulty, we were able to familiarize ourselves with ChocoPy's grammar, and we had to settle on how we were planning on implementing the translation from ChocoPy to Bril. At the recommendation of Professor Sampson, we focused on efforts on the AST translation, instead of actually building the AST from scratch, as this would be a really tedious, time consuming, and fairly non-trivial task to do. This allowed us to focus on the code generation phase of the compiler, which was our main goal. The next step in the design would be to decide how we were going to generating the AST from the ChocoPy program. Originally, we thought that the simplest solution would be to use the built in Python AST module to generate the AST from the ChocoPy program. However, we soon realized that there were small nuances between the ChocoPy grammar and the Python grammar, which made this approach infeasible. After some research, we found that there were multiple existing ChocoPy parser implemented in Python, which we were able to leverage to generate the AST for us. We ended up deciding to use "yangdanny97/chocopy-python-compiler" as our ChocoPy parser, as it was fairly well documented and seemed easy to use.

For the implementation, we directly called the "yangdanny97/chocopy-python-compiler" code to generate the AST. Luckily, the AST that was generated was basically in JSON format. As a result, we just had to hard code the translation between the ChocoPy AST nodes and the Bril instructions. Most notably, we created a `CodeState` class to keep track of temporary variables and variable types. This helps generate unique temporary variables names during the translation. The `translate_prog` function serves to handle function definitions, the ChocoPy main function body, as well as any global variable declarations. `translate_func_def` is responsible for mapping arguments and local variable definitions, translating statements, and adding a default return value if none is proveded. `translate_var_def` translates variable definitions, and `translate_expr` handles the translation of instructions, including literals, identifiers, binary operations, unary operations, and function calls. Finally, `translate_stmt` translates statements such as assignment, if statements, while loops, return statements, and expression statements including print. The main function orchestrates the overall AST translation, including calling "yangdanny97/chocopy-python-compiler" to create the ChocoPy AST, as well as calling the appropriate functions from above to generate the Bril code.

Finally, testing also played an important role in our implementation. We curated a test suite of 20 unit tests that covered the various features we implemented in our ChocoPy2Bril translator. Each test case consisted of a ChocoPy program, and we wrote a script that verified that the result of running the generated Bril code matched the ChocoPy program's output. This helped us ensure the correctness of our implementation and catch any bugs or edge cases. Our test suites were partially adapted/inspired from the test suites provided in the "yangdanny97/chocopy-python-compiler" repository. Initially, we wanted to directly use tests from "yangdanny97/chocopy-python-compiler". However, we soon realized that most of those tests were fairly complex and covered a lot of ChocoPy features that we were not planning on implementing. As a result, we had to edit them to fit our implemented features, as well as manually write some of our tests from scratch. Additionally, on the GitHub issue for this project, we were intially planning on using Turnt for testing. However, we soon realized that since the way that we were checking correctness/calling functions differed across test cases, it was simply easier to write our own testing script that handled each test case individually.

This is an example of one of our test cases, `is_even`:
```ChocoPy
def is_even(n: int) -> bool:
    half = n // 2
    return n == half * 2

x: int = 8
y: int = 5

is_even_x: bool = False
is_even_y: bool = False
is_even_x = is_even(x)
is_even_y = is_even(y)
print(is_even_x)
print(is_even_y)
```

When we run our ChocoPy2Bril translator on this test case, it generates the following Bril code:
```Bril
@is_even(n: int): bool {
  _temp0: int = id n;
  _temp1: int = const 2;
  _temp2: int = div _temp0 _temp1;
  half: int = id _temp2;
  _temp3: int = id n;
  _temp4: int = id half;
  _temp5: int = const 2;
  _temp6: int = mul _temp4 _temp5;
  _temp7: bool = eq _temp3 _temp6;
  ret _temp7;
}
@main: None {
  _temp0: int = const 8;
  x: int = id _temp0;
  _temp1: int = const 5;
  y: int = id _temp1;
  _temp2: bool = const false;
  is_even_x: bool = id _temp2;
  _temp3: bool = const false;
  is_even_y: bool = id _temp3;
  _temp4: int = id x;
  _temp5: bool = call @is_even _temp4;
  is_even_x: bool = id _temp5;
  _temp6: int = id y;
  _temp7: bool = call @is_even _temp6;
  is_even_y: bool = id _temp7;
  _temp8: bool = id is_even_x;
  print _temp8;
  _temp9: bool = id is_even_y;
  print _temp9;
  ret;
}
```

### What were the hardest parts to get right?
The hardest part of the project was definitely the debugging. It was often difficult to trace back errors in the generated Bril code to the corresponding ChocoPy AST nodes. This made it challenging to identify and fix bugs in our implementation. When the test results did not match, we had to analyze whether it was our code that was incorrect, or if there was an issue with the ChocoPy parser we were using. Additionally, there were moments where our Bril file translation would fail, and we had to inspect whether it was due to our translation logic or if there were limitations in the Bril language itself. 

Another difficulty in our project was definitely trying to generate our test suite. Since we were only implementing a subset of ChocoPy, we had to be careful to ensure that our test cases only used the features that we implemented. Additionally, we had to make sure that our test cases covered a wide range of scenarios and edge cases to ensure the correctness of our implementation. The whole process of curating the test suite took a significant amount of time and effort, but we also understand that it is a crucial part of the development process.

### Were you successful? (Report rigorously on your empirical evaluation.) 
Overall, we were fairly successful in implementing the ChocoPy2Bril translator. We were able to successfully translate the defined subset of ChocoPy features into Bril code. Our test suite consisted of 20 unit tests, and we were able to pass all of them, which gives us confidence in the correctness of our implementation. We tried to be thorough with our test suite, covering various scenarios and edge cases. However, we acknowledge that there may still be some edge cases that we did not cover, and there is always room for improvement in terms of performance and optimization. Nonetheless, we are satisfied with the overall success of our implementation and the results of our empirical evaluation.