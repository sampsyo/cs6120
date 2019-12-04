+++
title = "The Cult of Posits"
extra.author = {"Dietrich Geisler", "Edwin Peguero"}
+++

Computers are incapable of representing arbitrary real numbers exactly.
This is due to two insurmountable cardinality facts of real numbers:
- Uncountably Infinite Quantity: There are an infinite number of real numbers.
- Uncountably Infinite Precision: Some real numbers require infinite precision.
 
Since computers use a finite number of bits, computer architects must settle on capturing a finite number of real numbers at a finite level of precision.
Furthermore, since bit patterns are finite, a tradeoff must be made between the number of representable numbers and the level of precision.
This is done by fixing a *representation*, that is, a mapping between bit patterns and real numbers.

## The Floating Point Representation

The *floating point representation* is the most widely used representation.
Approximately, it represents numbers of the form `(-1^s) * 1.m * 2^e`, where `1.m`, the *mantissa*, and `e`, the *exponent*, are represented under the fractional and integer binary system, respectively, and `s` is a single bit denoting the sign of the represented number.
The design tradeoff between quantity and precision is captured by the number of bits dedicated to the mantissa and exponent, respectively.

In practice, however, the IEEE 754 floating point standard slightly modifies this scheme to account for two perceived limitations:
- Small Number Gap: there is a relatively large gap between the representation of the largest negative number, and the smallest positive number
    - To account for this, the numbers with the smallest exponent are *denormalized*. 
    This means that they are spread out linearly, rather than exponentially, between the largest negative and smallest positive numbers raised to the second largest exponent.

[//]: # (Note: this is a hyperlink used as a comment lol)
[//]: # (TODO: insert image of denormalized numbers here, such as from: http://www.toves.org/books/float/#s2.1 )
[//]: # (TODO: add images of piecewise floating point definition from https://posithub.org/docs/Posits4.pdf)

- Bogus Results: the result of overflow (e.g., dividing by a very small number) or partial functions applied to elements outside of their domains (e.g, division by zero) have no representation
    - The case of overflow is captured by the *positive and negative infinity* values, each represented by the bit pattern corresponding to an all ones exponent and all zeros mantissa, and differentiated by the sign bit.
    - The case of a non-result of a partial function is captured by the *NaN* value (meaning, "not a number"), represented by the various bit patterns with an all ones exponent and non-zero mantissa.

## The Posit Representation

The *posit representation* should be the most widely used representation.
The numbers represented by posits are similar to floating points, but differ by the introduction of a so-called *regime* term, as follows: 

````(-1^s) * 1.m * useed^k * 2^e````

`useed = 2^(2^es)`, a fundamental quantity in the theory of numerical representations, parametrized by the quantity `es`.

[//]: # (TODO: add images of piecewise posit definition from https://posithub.org/docs/Posits4.pdf)

In his [seminal paper](https://posithub.org/docs/Posits4.pdf), Gustafson explains the genius behind this design:
> The regime bits may seem like a weird and artificial construct, 
but they actually arise from a natural and elegant geometric mapping of binary integers to the projective real numbers on a circle.

> ... The value 2^(2^es) is called useed because it arises so often.

Fascinating. 

## Supremacy of Posits

The posit representation maps numbers around the topological circular loop in quadrants, as prophesied by several major religions.

[//]: # (insert image of circle with 4 cardinal points here from https://posithub.org/docs/Posits4.pdf)

At the heavenly North of the circle, symbolizing the Alpha and Omega, Our Father to which we solemly pray, lies the glorious positive and negative infinity.
At its opposite, the wicked, immoral South of the circle, lies nothing of value, the value `0`.
Meanwhile, on the earthly plane, God's children enjoy free will, where they choose between positive one at the East and negative one at the West.

The quadrants induced by these points are then symmetrically populated by the rest of the points. 
`useed` determines where the "center" of these quadrants resides as follows:

[//]: # (insert image of circle with useed values here from https://posithub.org/docs/Posits4.pdf)

Much like Adam and Eve did, `useed` determines how the quadrants in the circle are populated.
Positive values lie at the right of the circle, while negative values lie at the left, and reciprocal values reflect across the equator.

## Comparing Numerical Representation Comparison

# Qualitative Comparison

Unlike IEEE's extravagantly piecewise nature, posits opt for piecewise minimalism:
- Whereas there are two floating point representations of zero (both positive and negative), there is only one such posits representation: the all zero bit pattern.
- Further, whereas positive and negative infinity are distinctly represented as floating points, posits unite these values into one representation: the bit pattern with all zeros save for the first bit.
- Finally, posits do away with the `NaN` values that pollute the floating point numbers.
 

# Metric-based Comparison

[//]: # (TODO: fill in Gustafson's metric definitions and comparisons here, from http://www.johngustafson.net/pdfs/BeatingFloatingPoint.pdf)
Not settling for simple, qualitative comparision, Gustafson proposes a variety of metric-based comparisons for numerical representations.
He asks: given a represenation, how can we assess how accurately each number is represented?
To answer this, he proposes the *decimal accuracy* metric: the number of significant bits a real number in the canonical representation of .

Let us regard *accuracy* as the *inverse of error*. 
Then each error metric, that is, each way of considering the distance between a computed value and its correct value, induces an accuracy metric. 
Thankfully, Gustafson reminds us that the choice of error metric is obvious:
  > A “perfect spacing” of ten numbers between 1 and 10 in a real number system would be not the evenly-spaced counting numbers 1 through 10, 
  > but exponentially-spaced 1,10^(1/10),10^(2/10),...,10^(9/10),10.

Thus, the "position" of a (positive, non-zero) real number `n` is given by `p`, where `n = 10^(p/10)`,
and so the distance between `n` and `n'` is given by `| log_10(n / n') | = p - p'`. 
In other words, we compare the spacing of numbers according to the *decibel system*, the inverse of which yields the *perfect metric for accuracy*.
Finally, we glean the significant digits of the representation, the *decimal accuracy*, by taking the base 10 logarithm of accuracy.

Gustafson exhaustively plots the decimal accuracy of each representation in 8-bits, demonstrating posit supremacy:

[//]: # (TODO: add image of decimal accuracy plot here)

We make two observations from this plot:
First, posits distribute decimal accuracy symmetrically across representations, while floating points fail to deliver at larger numbers, which are ignored in favor of `NaN`.
Furthermore, posits favor decimal accuracy at the meat of the number line, a sizable domain around 0, whereas floating points over-emphasizes the ends of the number line.
Finally, the posit representation casts the light of representation across a larger domain; its *dynamic range* exceeds that of floating points.

But how does error propagate with successive operations?
> Note that the values represented by posits represent over two decades more *dynamic range* than the floats, 
> as well as being as accurate or more for all but the values where floats are close to underflow or overflow.

(note that floating points have better 

- Dynamic Range
- single argument operation comparisons (sqrt, exp, etc.)
- two argument operation accuracy and closure


## Evaluation

We compare the accuracy of 32-bit floating point and posit representation by comparing their accuracy under a variety of benchmarks.
In each benchmark, we express real number calculations in terms of operations over 64-bit `double`s.
In other words, the `double` type serves as the "oracle" baseline from which we compute accuracy.
Each `double` benchmark is then compiled to LLVM IR, upon which we apply a `float` LLVM pass and a `posit` LLVM pass respectively to generate `float` and `posit` benchmarks.
The values produced from these two benchmarks carry the accumulated error arising from the inaccuracy of the particular 32-bit representation, which compounds with successive operations.
Finally, we derive accuracies by comparing to the value of the `double` baseline benchmark, and use these as metrics to compare the accuracy of the two representations.

Although it is not true that the `double` type has perfect accuracy, since no finite representation is, we assume that the accumulated error in `double` benchmarks will be truncated or rounded off when comparing with the less precise 32-bit floating representations.
In other words, we assume that the precision of the most significant, inaccurate digit in a 64-bit `double` benchmark result cannot be represented by a 32-bit `float` or `posit`.
Although this assumption does not generally hold, it usually does, and it helps streamline the implementation of the different benchmarks.

# LLVM `float` pass

The LLVM `float` pass generates a `float` benchmark from a `double` benchmark by casting all `double` operations to `float` operations and all `double` operands to `float` operands.
Conversion of `double` operations to `float` operations happens "for free", 
In so doing, the resulting value carries the accumulated error characteristic of the successive `float` operations on 32-bit `float`.

# LLVM `posit` pass

The LLVM `posit` pass operates analogously: it converts `double` operations and operands to the `posit` versions.
This conversion is limited by the availability of `posit` operation implementations.
We draw `posit` addition, subtraction, multiplication, and division implementations from the [Cerlane Leong's SoftPosit repository](https://gitlab.com/cerlane/SoftPosit-Python).

### Benchmarks

[//]: # (TODO: figure out this part)

## Conclusion

[//]: # (TODO: do this after benchmarks)
