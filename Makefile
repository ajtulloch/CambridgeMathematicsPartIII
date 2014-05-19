DBOX=~/Dropbox/Sharing/CambridgeSummaries

summaries:
	find . -iname 'master.pdf' | grep Summary | xargs -I{} gcp --parents {} $(DBOX)
