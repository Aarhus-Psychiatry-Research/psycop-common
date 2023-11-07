pr:
	make merge-main
	inv pr

merge-main:
	git fetch
	git merge --no-edit origin/main

enable-automerge:
	gh pr merge --auto --squash --delete-branch

squash-from-parent:
	git fetch
	git reset $$(git merge-base origin/main $$(git rev-parse --abbrev-ref HEAD)) ; git add -A ; git commit -m "Squash changes from parent branch"

create-random-branch:
	@git checkout -b "$$(date +'%y_%m_%d_%H')_$(shell cat /dev/urandom | env LC_ALL=C tr -dc 'a-z' | fold -w 5 | head -n 1)"

grow:
	@echo "––– Growing into a new branch 🌳 –––"
	make create-random-branch
	make squash-from-parent

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