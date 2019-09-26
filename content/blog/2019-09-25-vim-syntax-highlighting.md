+++
title = "Bril Syntax Highlighting for Vim"
extra.author = "Edwin Peguero"
+++

### Bril Syntax Highlighting
This project aimed to provide syntax highlighting for Bril in the vim text editor, with the goal of learning about the implementation process underlying this ubiquitous category of tools. Until now I've taken for granted this tooling across various editing environemnts and programming languages; so I felt that the ability to support my language design efforts might prove useful and interesting.

As the project evolved, I quickly discovered the task of *syntax highlighting* to be more open-ended than expected. Ideally, we want the appearance of program text to reflect syntactic structure, but:
- Maintaining a constantly changing syntax tree for an entire program can be slow, and
- The nature of editing is that program text does not always represent a well-formed syntax tree.

This fundamental limitation is acknowledged at the start of the documentation for `syntax`, the collection of syntax highlighting commands in vim:
> Syntax highlighting enables Vim to show parts of the text in another font or color. Those parts can be specific keywords or text matching a pattern.  Vim doesn't parse the whole file (to keep it fast), so the highlighting has its limitations.  Lexical highlighting might be a better name, but since everybody calls it syntax highlighting we'll stick with that.

Even though *syntax* highlighting implies the output of a parsing operation, the reality is closer to *lexing*. However, vim's powerful regular expressions, in conjunction with the features available through some `syntax` commands, facilitate highlighting that appears more complex than simply highlighting tokens.

#### Anatomy of a Vim Syntax Highlighter
The basic structure of a vim syntax highlighter is designed to separate the concerns of textual appearance from textual extraction of syntactic units. 
A colorscheme will map a language's *syntax groups* to a set of generic *highlight groups* provided by vim. 
Each highlight group is named after a generic syntax unit, such as `Comment` or `Identifier`, and defines its appearance. 
Thus, the development efforts of language and colorscheme designers are independent, as seen in the implementation of `bril-syntax`: 
```
highlight default link brilComment Comment
highlight default link brilLabel Label
highlight default link brilVariable Identifier
highlight default link brilMain Function
highlight default link brilType Type
highlight default link brilValueOp Operator
highlight default link brilEffectOp Keyword
highlight default link brilNumber Number
highlight default link brilBool Boolean
highlight default link brilCondVariable Boolean
```
My task was simply to define each of the above `bril-` syntax groups.

It's interesting to observe that although this scheme works well for most programming languages, it does not work for any arbitrary formal language, since the baseline set of highlight groups makes an assumption about the 'top' level of syntactic categories. From the `syntax` docs:
> ... a syntax group and a highlight group are similar. For a highlight group you will have given highlight attributes. These attributes will be used for the syntax group with the same name.

To resolve this, one might create new highlight groups, as follows:
```
highlight MyHighlightGroup gui=bold ctermbg=NONE cterm=bold ...
```
However, doing so conflates the roles of language designer and colorscheme designer. To maintain this separation of concerns for more exotic languages, a new set of highlight groups would need to be exposed for colorscheme designers.

Since Bril's syntax fits nicely into the base set of highlight groups, this was not a problem. 
I'll admit, though, that I really wanted to italicize at least one syntax group, and although it's possible to minimally override a colorscheme in this way, any implementation is error prone, possibly version-dependent, and hard to understand and maintain (as is much of **vimscript**).
Thus, I focused my efforts on designing a the needed collection of `bril-` syntax groups.

#### Syntax Groups and Regions
The most basic syntax group is a keyword, defined using the `syntax keyword` command.
It takes two arguments: a name for the syntax group and a set of language keywords.
For example, value ops are defined as follows (ignore the `contained` option, for now):
```
syntax keyword brilValueOp contained
  \ id
  \ const
  \ add
  ...
```

At the next level of generality, we can define syntax groups using regular expressions via the `syntax match` command. For example, bril comments are defined as follows:
```
syntax match brilComment "\#.*$"
```

Vim provides one final level of generality with the notion of *syntax regions*.
A syntax region is a region of text, delimited by regular expressions on both sides:
```
syntax region regionName start=startRegexp end=endRegexp contains=synGroup1,synGroup2,...

```
When the *start* expression is detected, only those syntax groups *contained* in a region are checked. 
The `contained` option removes top-level visibility from a syntax group, so that they are only parsed when their parent syntax region is parsed. 
Together, these two mechanisms allow for hierarchical parsing of 'syntax-tree-like' syntax groups. 
Besides its effectiveness as a design pattern for organizing syntax groups, regions made it easier to identify the branch condition as a `brilCondVariable`:
```
syntax region brilBranchInstr start='br' end=';' oneline contained contains=brilCondVariable,brilVariable,brilEffectOp
```

#### Local Parsing via `sync` points
When scrolling through a file or making an edit, Vim needs to figure out the most fitting syntax groups in the corresponding line. Since syntax groups may lie within a syntax region, Vim needs to find the most accurate *syntax state* for the new line:
>Vim wants to be able to start redrawing in any position in the document.  To make this possible it needs to know the syntax state at the position where redrawing starts.

A robust, but slow syntax highlighter can recalculate syntax state across the entire file as needed. 
To improve performance, Vim introduces mechanisms for defining a *sync point* around which syntax state is remembered.
Vim allows for defining a sync point relative to screen and cursor details, including the cursor line number, the lines that have currently been drawn on screen, and user defined look-behind parameters. 
Additionally, Vim provides a mechanism for locally 'guessing' the current syntax region using regular expressions as hints. Although, this requires some parsing several lines, it might be preferable than parsing the entire file.

For now, `bril-syntax` uses the `syntax sync fromstart`, which, as the name implies, sets the sync point at the start of the file. Thus, the entire file is parsed with each new line.

#### Evaluation
A syntax highlighter can be evaluated both quantitatively and qualitatively.
On the first front, it must be performant enough to not slow down the development workflow.
On the qualitative side, is useful while typing, 


### Implementation

### Design Challenges and Considerations

### Evaluation

