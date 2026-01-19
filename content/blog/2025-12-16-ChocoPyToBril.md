### What was the goal
The goal was to implement a frontend that translates ChocoPy to Bril, utilizing an existing ChocoPy parser and generate correct Bril code for a representative subset of the language. Features that I would cover concerning ChocoPy's language will be:
- Types: Integers, Booleans
- Basic Operations: + - * // % < <= > >= == != and or not
- Local variable assignment and declaration
- If/else and while loops
- Function definition, argument passing, return statements

### What did you do?
I implemented what my goal was, covering almost all of the features I stated concerning ChocoPy’s language, except for strings and characters. Overall, I am very satisfied with both how I implemented my solution and how it works. Furthermore, I also implemented modulo by making a compile_modulo command that puts sequential operations that function as the modulus operator.
An example of what happens is shown below:

Input: 
```python
x = 1 % 2
print(x)
```
Output:
```json
@main {
  t0: int = const 1;
  t1: int = const 2;
  t2: int = div t0 t1;
  t3: int = mul t2 t1;
  t4: int = sub t0 t3;
  x: int = id t4;
  print x;
  ret;
}
```
### What were the hardest parts to get right?
The hardest parts for me definitely had to be function definition and while loops. I struggled for a bit to understand completely how functions were placed in bril JSON, as well as how to translate them correctly. The reason I was able to implement both was taking a whiteboard and breaking both of them down structurally, and then piecing them together in my code one attribute by attribute. In particular, one menacing error that haunted me throughout this project was that "1 function required". It took me several hours to debug this error as it was not alot to go off of. After paintstakingly comparing bril JSONs to the benchmarks in the bril repo, I realized that my error lied in how I had structured my call op, where it had the func attribute instead of funcs, and had an array. I was confused on why it had multiple funcs as I have never seen a bril file that actually makes the json use more than one function within the call, but after fixing it all was well. It was showing as 
```json
        {
          "op": "call",
          "func": "start",
          "args": [
            "t0"
          ]
        },
```
When it shouldve been showing as 
```json
        {
          "op": "call",
          "funcs": [
            "start"
          ],
          "args": [
            "t0"
          ]
        },
```

### Were you successful?
I realized that Bril does not support strings, and that implementing these in Bril would be essentially “hardcoded". I originally thought that regardless of this I could make a string table, storing the actual values correlated to an integer id that would be stored in bril. However, I realized attempting to print a string would not work as bril would not support printing a type that doesn’t exist in itself (in this case it would be a buggy sequential printing of several characters that would not look well). In order to implement strings in an non-buggy way, I would need to implement several more features in Bril like lists, as well as make an entire wrapper around bril. As a result, I had decided not to implement characters and strings. I was overall successful as all my test cases passed. For my test cases I used all the test cases that utilized only the features I had covered from [here](https://github.com/gabe-raulet/chocopy2bril/tree/master/examples), as well as base test cases that were used to work out and develop my solution. The tests did have to be rewritten slightly to be checked (where I would remove the main function’s input, and simply have it call the main with set variables), but other than that they were the same as the original. In order to test, I checked the difference in output between executing with CPython and executing with my compiler implementation. There is a run_diff.sh script that executes this for every test in the test folder. I made 20 tests, and they all had the exact same output as they would in python. Originally, in my proposal I had stated 20 tests from a directory. However, I realized, funnily enough, that this directory didn’t have 20 test cases and that the amount of files I saw in the directory was inflated due to .opt.bril, _hand_impl, and .compiled.opt files. Due to this, I fetched more test cases from online, linked [here](https://github.com/gabe-raulet/chocopy2bril/tree/master/examples). Funnily enough, for booleans there was also a mismatch because Python has “True” and brili has “true”, which I thought was similar enough to be negligible.  
