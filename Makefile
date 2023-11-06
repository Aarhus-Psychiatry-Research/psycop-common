pr:
	inv pr

squash-from-parent:
	git fetch
	git reset $$(git merge-base origin/main $$(git rev-parse --abbrev-ref HEAD)) ; git add -A ; git commit -m "Squash changes from parent branch"

create-random-branch:
	@git checkout -b "$$(date +'%y_%m_%d_%H')_$(shell cat /dev/urandom | env LC_ALL=C tr -dc 'a-z' | fold -w 5 | head -n 1)"

grow:
	@echo "––– Growing into a new branch 🌳 –––"
	make create-random-branch
	make squash-from-parent

g:
	make grow

gpr:
	make pr
	make g