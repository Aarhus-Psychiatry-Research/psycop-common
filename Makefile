warning:
	@echo "🚨🚨🚨 WARNING: This Makefile is deprecated. Please use invoke instead. It will be removed in the future 🚨🚨🚨"

pr:
	make warning
	make merge-main
	inv qpr

push:
	make warning
	@echo "\n––– Pushing to origin/main –––"
	@git push --set-upstream origin HEAD
	@git push
	@echo "✅✅✅ Succesful push! ✅✅✅"

merge-main:
	make warning
	@echo "\n––– Merging main –––"
	git fetch
	git merge --no-edit origin/main
	@echo "✅✅✅ Succesful merge! ✅✅✅"

enable-automerge:
	make warning
	gh pr merge --auto --delete-branch

create-random-branch:
	make warning
	@git checkout -b "$$(date +'%y_%m_%d_%H')_$(shell cat /dev/urandom | env LC_ALL=C tr -dc 'a-z' | fold -w 5 | head -n 1)"

grow:
	make warning
	@echo "\n––– Growing into a new branch 🌳 –––"
	make create-random-branch
	make merge-main

lint:
	make warning
	@echo "\n––– Linting –––"
	pre-commit run --all-files

#################
# Short aliases #
#################
mm:
	make warning
	make merge-main

g:
	make warning
	make grow

gpr:
	make warning
	make pr
	make g