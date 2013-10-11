all:
	latexmk -pdf master.tex

cont:
	latexmk -pdf -pvc master.tex

view:
	open master.pdf

clean:
	rm -rf *.dvi *.aux *.fls *.fdb_latexmk *.log *.out *.pdf _region_.tex auto

publish:
	rm -rf PDFs
	mkdir PDFs
	for f in **/*.pdf; do \
		fnew=`echo $$f | sed 's|/|_|'`; \
		cp $$f PDFs/$$fnew; \
	done
