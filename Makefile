DBOX=~/Dropbox/Sharing/CambridgeSummaries

summaries:
	find . -iname 'master.pdf' | xargs -I{} gcp --parents {} $(DBOX)
