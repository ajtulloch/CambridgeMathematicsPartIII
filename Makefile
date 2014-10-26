DBOX=~/Dropbox/Sharing/CambridgeSummaries

summaries:
	find . -iname 'master.pdf' | xargs -I{} gcp --parents {} $(DBOX)

collate:
	mkdir -p scratch/
	find . -iname 'master.pdf' | xargs -I{} cp $(echo "{}" | tr / -) scratch/
