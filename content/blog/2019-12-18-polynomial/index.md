+++
title = "Code Generation for Polynomial Multiplication" 
extra.author = "Wen-Ding Li" 
extra.bio = """ 
  [Wen-Ding Li](https://www.cs.cornell.edu/~wdli/) is a first year PhD student interested in high performance computing.
""" 
extra.latex = true
+++


High performance cryptography algorithm implementation has always been an active
research topic among crypto communities, because this software needs to be
deployed widely in different situations where performance can become crucial.
Recently, a new kind of crypto system based on ideal lattices has attracted a lot
of attention and can potentially be adopted as the new standard. The primary
computation for ideal lattice crypto systems are polynomial multiplication. In
this project, we want to explore how to generate high performance polynomial
multiplication for these kind of crypto systems. 

## Motivation

### Post-Quntum Cryptography

A lot of public-key cryptography systems we used today relying on primitives
such as RSA and elliptic curve cryptography which are vulnerable under the quantum computer
attacks.  As people now suggest that the threat of quantum computer might
not be that far in the future, cryptography researchers have been studying
so-called post-quantum cryptography (PQC), the kind of crypto primitives that we
believe can withstand the quantum attack.  There are a lot of post-quantum
cryptosystems being proposed recently and post-quantum cryptography researchers
now are actively trying to analyze the security and performance of these
systems.

### PQC Candidates

Although the question about how far are we from building a quantum computer that
can break the secure communication today is still in heated debate, we now need
a guideline for people who want to migrate to choose among all those
post-quantum crypto systems proposed in the literature so far.  National
Institute of Standards and Technology (NIST) started a
[process](https://csrc.nist.gov/Projects/post-quantum-cryptography) to standardize
the post-quantum crypto in 2017 and as of today (Dec 2019), there are still 17
candidates for public-key encryption and 9 candidates for digital signatures in
[round 2](https://csrc.nist.gov/Projects/post-quantum-cryptography/round-2-submissions).
When choosing which systems will become the standard, security is obviously the
priority. However, the performance of the systems is also very important because
once chosen as part of the standard, it would very likely to be implemented in
popular libraries such as OpenSSL and deployed to millions of devices and servers
and have impact on billions of users.

### High Performance PQC Implementation 

Thus, it is very important to analyze the performance of these systems, and
know how the most optimized version of the code performs on each platform.
Yet, turning an algorithm into highly optimized code is not an easy job. To
make things even harder, now there are many different crypto systems and each of
them have different level of security parameters to select.  Hand crafting
optimized code for every one of them and analyze their performance is a lot of
work and it is also not a fair comparison among different systems if some
researchers just don't have the resources to invest a lot of time on crafting
that highly optimized assembly code. Here, we hope to start addressing this
problem by creating a code generation system whose its goal is to create high
performance polynomial multiplication code commonly used in a lattice-based
post-quantum crypto systems.

### High Performance Code Generation

Generating high performance code to run on different hardware has been studied
for decades. The signal processing community have been studied how to generate high
performance FFT implementations, like [FFTW](http://www.fftw.org/) and
[Spiral](https://www.spiral.net/index.html). In recent years, image
processing and machine learning communities also propose several frameworks to
generate high performance code, such as on the
[Halide](https://github.com/halide/Halide), [Loopy](https://github.com/inducer/loopy) and
[TVM](https://github.com/apache/incubator-tvm). 

### Spiral: FFT code generator
Spiral is a state of the art FFT code generator. FFT can be expressed as product of sparse matrices. 
Spiral's framework includes an DSL called SPL language that can be used to 
described FFT's algorithm transformation.
Currently there are around 20 break-down rules in Spiral that can be used to
break down a transformation into a series of transformations. Using different
rules to decomposed the matrix, we can get different implementation of the FFT
algorithm. By trying out different combination of rules, it can find the best
implementation on the target platform. 

## Efficient Polynomial Multiplication Algorithm

Inspired from Spiral, I thought that we could similarly write down the rules to 
break down polynomial multiplication, and we can use these rules to break down polynomial
multiplication into smaller ones. As we can see in this section, there are
different rules we can use to recursively break down the polynomial.


### Schoolbook Multiplication
Suppose we want to multiply two polynomials $f(x)$ and $g(x)$, where
  $$
  f(x) = a_1(x) t + a_0(x)
  $$
  $$
  g(x) = b_1(x) t + b_0(x)
  $$

  Then schoolbook method is to compute 
  $$
  f(x)g(x) = c_2(x) t^2 + c_1 t + c_0(x)
  $$
  where
  $$ c_2(x) = a_1(x)b_1(x) $$
  $$ c_1(x) = a_1(x)b_0(x) + a_0(x)b_1(x)$$
  $$ c_0(x) = a_0(x)b_0(x) $$.
  It can break down a large polynomial multiplications into 4 smaller polynomial
  multiplications.

### Karatsuba
We can break the polynomial $f(x)$ and $g(x)$ to higher degree part
and lower degree part as follows:

  $$
  f(x) = a_1(x) t + a_0(x)
  $$
  $$
  g(x) = b_1(x) t + b_0(x)
  $$
  , where $t = x^{\frac{n}{2}}$. 
  Then the multiplication can be calculated as follows:

  $$
  f(x)g(x) = c_2(x) t^2 + c_1 t + c_0(x)
  $$

  where 
  $$ c_2(x) = a_1(x)b_1(x) $$
  $$ c_0(x) = a_0(x)b_0(x) $$

  $$
  c_1(x) = (a_1(x) + a_0(x)) (b_1(x) +b_0(x)) -c_0(x)-c_2(x)
  $$

  It uses 3 multiplications while the schoolbook method would use 4.
  The idea can be further generalized to a k-way split, and it is called Toom-k.

### Toom-k

Lets take Toom-3 as an example. First we divide the polynomial $f(x)$ and $g(x)$ into 3 parts.

$$
f(x) = a_2(x) t^2 + a_1(x) t + a_0(x)
$$

$$
g(x) = b_2(x) t^2 + b_1(x) t + b_0(x)
$$

where $t = x^{\frac{n}{3}}$. 
Similarly to Karatsuba method, in this case, the number of
multiplication can be reduced to 5, instead of 9 in the case of schoolbook
method and asymptotically better than both Karatsuba method and schoolbook
method. This method can be generalized to $k$-way split.

### Unable to decouple the algorithm from the order of computation
However, I am unable to find a way to decouple the above algorithm from the
order of computation while still be easy to configure and be useful. As a
result, it makes this project unsuccessful.

## Implementation

The goal of this project is to build a system whose input is n, the degree of
the input polynomial, and output a optimized AVX2 code. To make this happen, we
need to at least take into account different algorithms and different order of
computation. Without the ability to specify the order of computation easily, I
do not accomplish the goal I set out to do. Nevertheless, I still implemented
different algorithm as the break down rules and generate C++ code with AVX2 for
fixed size polynomial multiplication. For each fixed size polynomial
multiplication, the code is completely unroll and the user can specify a set of rules
at each recursion level. The code generator and generated code can be found 
in this [repository](https://github.com/xu3kev/polymul_gen).  
Some benchmark of the cycle count is report as follows. The testing machine is
Intel Xeon Gold 6136 CPU @ 3.00GHz.

 n   | cycle count
-----|----
 32  |4930
 64  |13810
128  |36760
256  |142038

### Conclusion

The goal of this project is to incorporate the idea from other code generators
like Spiral and Halide, in order to generate high performance code for certain
fixed size polynomial multiplication. However, this attempt is unsuccessful. Although
polynomial multiplication is also an very structural computation, it is not like FFT
which is a linear transform, and thus not easy to directly borrow ideas from
Spiral framework. Despite the futile attempt, I still think it is an interesting
and important problem worth thinking about.

