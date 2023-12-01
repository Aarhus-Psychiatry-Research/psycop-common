@task
def create_pr(c: Context):
    """
    Created a PR, does not run tests or similar
    """
    try:
        pr_result: Result = c.run(
            "gh pr view --json url -q '.url'",
            pty=False,
            hide=True,
        )
        print(f"{msg_type.GOOD} PR already exists at: {pr_result.stdout}")
    except Exception:
        branch_title = c.run(
            "git rev-parse --abbrev-ref HEAD",
            hide=True,
        ).stdout.strip()
        preprocessed_pr_title = branch_title.split("-")[1:]
        preprocessed_pr_title[0] = f"{preprocessed_pr_title[0]}:"
        pr_title = " ".join(preprocessed_pr_title)

        c.run(
            f'gh pr create --title "{pr_title}" --body "Automatically created PR from invoke" -w',
            pty=NOT_WINDOWS,
        )
        print(f"{msg_type.GOOD} PR created")


def update_pr(c: Context):
    if not on_ovartaci():
        echo_header(f"{msg_type.COMMUNICATE} Syncing PR")
        # Get current branch name
        branch_name = Path(".git/HEAD").read_text().split("/")[-1].strip()
        pr_result: Result = c.run(
            "gh pr list --state OPEN",
            pty=False,
            hide=True,
        )

        if branch_name not in pr_result.stdout:
            create_pr(c)


@task
def create_pr_from_staged_changes(c: Context):
    start_branch_name = c.run(
        "git rev-parse --abbrev-ref HEAD",
        hide=True,
    ).stdout.strip()

    pr_title = input("Enter PR title: ")
    branch_title = re.sub(r"[\-\s\:\,]", "_", pr_title).lower()
    c.run("git checkout main")
    c.run(f"git checkout -b {branch_title}")
    c.run(f"git commit -m '{pr_title}'")
    push_to_branch(c)
    update_pr(c)

    checkout_previous_branch = input("Checkout previous branch? [y/n]: ")
    if checkout_previous_branch.lower() == "y":
        c.run(f"git checkout {start_branch_name}")
