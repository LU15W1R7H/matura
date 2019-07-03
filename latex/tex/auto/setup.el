(TeX-add-style-hook
 "setup"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("fontenc" "T1") ("babel" "german") ("SIunits" "amssymb") ("biblatex" "backend=biber" "citestyle=authortitle") ("footmisc" "bottom")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "fontenc"
    "babel"
    "environ"
    "amsmath"
    "amsfonts"
    "amssymb"
    "SIunits"
    "mathtools"
    "esvect"
    "bm"
    "xfrac"
    "biblatex"
    "csquotes"
    "graphicx"
    "wrapfig"
    "tikz"
    "tikzpagenodes"
    "pgfplots"
    "xcolor"
    "subfiles"
    "geometry"
    "fancyhdr"
    "footmisc"
    "background"
    "url"
    "titlesec")
   (LaTeX-add-bibliographies
    "references")
   (LaTeX-add-xcolor-definecolors
    "bgcolor"
    "textcolor"))
 :latex)

