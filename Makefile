.PHONY: site serve deploy

# Clean build.
site:
	rm -rf public
	zola build --drafts

serve:
	zola serve --drafts

# Deployment.
RSYNCARGS := --compress --recursive --checksum --itemize-changes \
	--delete -e ssh --perms --chmod=Du=rwx,Dgo=rx,Fu=rw,Fog=r
DEST := cslinux:/courses/cs6120/2023fa/site
deploy: site
	rsync $(RSYNCARGS) ./public/ $(DEST)
