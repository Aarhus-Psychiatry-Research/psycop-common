pr:
	make merge-main
	inv qpr

push:
	@echo "\n––– Pushing to origin/main –––"
	@git push --set-upstream origin HEAD
	@git push
	@echo "✅✅✅ Succesful push! ✅✅✅"

merge-main:
	@echo "\n––– Merging main –––"
	git fetch
	git merge --no-edit origin/main
	@echo "✅✅✅ Succesful merge! ✅✅✅"

enable-automerge:
	gh pr merge --auto --delete-branch

create-random-branch:
	@git checkout -b "$$(date +'%y_%m_%d_%H')_$(shell cat /dev/urandom | env LC_ALL=C tr -dc 'a-z' | fold -w 5 | head -n 1)"

grow:
	@echo "\n––– Growing into a new branch 🌳 –––"
	make create-random-branch
	make merge-main

lint:
	@echo "\n––– Linting –––"
	pre-commit run --all-files

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