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

## Method 

### Schoolbook Multiplication

This is the straight forward basic method.

### Karatsuba
We break the polynomial $f(x)$ and $g(x)$ to high and low two part.


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
  The idea can be further generalize to k-way split, and it is called Toom-k.

### Toom-k

Lets take Toom-3 as an example. First we divide the polynomial $f(x)$ and $g(x)$ into 3 parts.

$$
f(x) = a_2(x) t^2 + a_1(x) t + a_0(x)
$$

$$
g(x) = b_2(x) t^2 + b_1(x) t + b_0(x)
$$

where $t = x^{\frac{n}{3}}$. 


Now, we substitute t with values: $0$, $1$, $-1$, $2$, and $\inf$.
and we let $f(x)g(x) = c_4(x)t^4 + c_3(x)t^3 + c_2(x) t^2 + c_1(x) t + c_0(x)$
We get the following table:

|  $t=$    |                                                      |                          |
|---------|------------------------------------------------------|------------------------------------------------|
|$ 0 $  |$a_0(x)b_0(x)$                                        |$=c_0(x)$                                        |
|$ 1 $  |$(a_2(x)+a_1(x)+a_0(x)) (b_2(x)+b_1(x)+b_0(x))$       |$=c_4(x) +c_3(x)+c_2(x)+c_1(x)+c_0(x)  $         |
|$ -1$  |$(a_2(x)-a_1(x)+a_0(x)) (b_2(x)-b_1(x)+b_0(x))$       |$=c_4(x) -c_3(x)+c_2(x) -c_1(x)+c_0(x)  $        |
|$ 2 $  |$(4a_2(x)+2a_1(x)+a_0(x)) (4b_2(x)+2b_1(x)+b_0(x))$   |$=16c_4(x) +8c_3(x)+4c_2(x)+2c_1(x)+c_0(x)  $    |
|$\inf$ |$a_2(x)b_2(x)$                                        |$=c_4(x)$                                        |

The second column is the 5 multiplications we perform, and then we can see that
we can solve an linear systems to get $c_4,c_3,c_2,c_1,c_0$ and mainly only use
addition, subtraction and constant multiplication. In this case, the number of
multiplication is 5, instead of 9 in the case of schoolbook method. We here
evaluate at 5 points, but this method can be generalized to $k$-way split evaluate
on $2k-1$ points.

### FFT

As we are dealing with polynomial with small to medium degree, we do not
consider the FFT method here. Yet it will be interesting to combine this with
above methods like in this [paper](https://eprint.iacr.org/2018/995).

## Implementation
We aim to generate polynomial multiplication over integer. We adapt from the 
NTRU crypto system [repository](https://github.com/joostrijneveld/NTRU-KEM).
We apply karatsuba and Toom-3 recursively and generate assembly code with AVX2
instructions.
