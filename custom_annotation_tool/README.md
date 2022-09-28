# Image Viewer

The original bounding boxes looked like this
`"(1, 2, 3, 4)"`
We can't easily parse this with just splitting operations.
We can use the regex replace mode in Notepad++ to circumvent this:
Find: `"((\d+), (\d+), (\d+), (\d+))"`
Replace: `$1-$2-$3-$4`

Currently the output image size is fixed, it can be changed along with the names of the folders and the image file types in lines 75-79 of viewer.html.

For running the viewer, run any kind of http server in the same directory as viewer.html (i.e. `python -m http.server`) and open the site in your browser (with python, thats http://localhost:8000).
