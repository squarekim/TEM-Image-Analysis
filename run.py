from tem_analyzer.deps import check

check()  # explain missing packages before Qt or OpenCV are imported

from tem_analyzer.gui import main  # noqa: E402

main()
