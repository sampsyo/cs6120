CS 6120
=======

This is the [website][cs6120] for a new grad course at Cornell on compilers.
It uses [Zola][].

[zola]: https://www.getzola.org
[cs6120]: https://www.cs.cornell.edu/courses/cs6120/2019fa/


Adding Blog Posts
-----------------

To add a blog post (which you must do for discussion leading and project reports), use a [pull request][pr].

You'll want to create a text file in the `content/blog/` directory with your new post.
Use a filename like `YYYY-MM-DD-title.md`, where the date is the discussion day or the project deadline and the title is up to you.
Include Zola-style "[TOML][] front matter" at the top, which looks like this:

    +++
    title = "Welcome to CS 6120!"
    [extra]
    bio = """
      Grace Hopper made the first compiler. [Adrian Sampson](https://www.cs.cornell.edu/~asampson/) is an assistant professor of computer science, so that's pretty cool too I guess.
    """
    [[extra.authors]]
    name = "Adrian Sampson"
    link = "https://www.cs.cornell.edu/~asampson/"  # Links are optional.
    [[extra.authors]]
    name = "Grace Hopper"
    +++

List all the authors of your post.
Include a link to your homepage if you have one, but it's optional.
Also write a short bio for yourselves (using [Markdown][]), which will appear at the bottom of the post.
Then, the rest of the text file is the Markdown text of your blog post.

If you want to use math in your blog post, put `latex = true` in your `[extra]` section to enable [KaTeX][]. Then you can use `$\pi$` for inline math and `\[ e^{i\pi} = 0 \]` for display math.

You can preview your writing with any Markdown renderer.
To see what it will look like when published, [install Zola][zola-install] and type `zola serve` to preview the entire site.

[pr]: https://help.github.com/en/articles/about-pull-requests
[toml]: https://github.com/toml-lang/toml
[markdown]: https://daringfireball.net/projects/markdown/
[zola-install]: https://www.getzola.org/documentation/getting-started/installation/
[katex]: https://katex.org
