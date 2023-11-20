pr:
	make merge-main
	inv qpr

push:
	@echo "––– Pushing to origin/main –––"
	@git push --set-upstream origin HEAD
	@git push

create-pr:
	@echo "––– Creating PR –––"
	@gh pr create --title "$$(git rev-parse --abbrev-ref HEAD | tr -d '[:digit:]' | tr '-' ' ')" --body "Auto-created" -w || true

merge-main:
	git fetch
	git merge --no-edit origin/main

enable-automerge:
	gh pr merge --auto --delete-branch

create-random-branch:
	@git checkout -b "$$(date +'%y_%m_%d_%H')_$(shell cat /dev/urandom | env LC_ALL=C tr -dc 'a-z' | fold -w 5 | head -n 1)"

grow:
	@echo "––– Growing into a new branch 🌳 –––"
	make create-random-branch
	make merge-main

#################
# Short aliases #
#################
mm:
	make merge-main

g:
	make grow

gpr:
	make pr
	make g