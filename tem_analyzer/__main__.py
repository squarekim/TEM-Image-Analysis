from .deps import check

check()  # explain missing packages before Qt or OpenCV are imported

from .gui import main  # noqa: E402

main()
