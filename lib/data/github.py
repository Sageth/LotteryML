import os
import git
import argparse
from github import Github


# Try to import the dotenv library. If not found, a message will be printed.


class GitHubAutoMerge:
    """
    A class to automate GitHub repository actions: committing staged changes,
    pushing, creating a pull request, and merging it.

    It requires a Git repository to be initialized and a GitHub Personal
    Access Token (PAT) with appropriate permissions (repo scope).
    """

    def __init__(self, repo_path, github_pat, github_owner=None, github_repo_name=None, remote_name='origin'):
        """
        Initializes the class with the repository path and GitHub PAT.

        Args:
            repo_path (str): The local path to the Git repository.
            github_pat (str): The GitHub Personal Access Token.
            github_owner (str, optional): The owner of the GitHub repository. Defaults to None.
            github_repo_name (str, optional): The name of the GitHub repository. Defaults to None.
            remote_name (str): The name of the remote to push to (default is 'origin').
        """
        self.repo_path = repo_path
        self.github_pat = github_pat
        self.remote_name = remote_name
        self.repo = None
        self.g = None
        self.remote_url = None
        self.github_owner = github_owner
        self.github_repo_name = github_repo_name

        try:
            # Initialize GitPython repository object
            self.repo = git.Repo(self.repo_path)
            # Initialize PyGithub client
            self.g = Github(self.github_pat)

            # If owner and repo name are not provided, try to get them from the remote URL
            if not self.github_owner or not self.github_repo_name:
                self.remote_url = next(
                    (remote.url for remote in self.repo.remotes if remote.name == self.remote_name), None
                )
                if not self.remote_url:
                    raise ValueError(f"Remote '{self.remote_name}' not found.")

                # Parse the remote URL to get the repository name and owner
                # Handles both SSH and HTTPS URLs
                if self.remote_url.endswith('.git'):
                    self.remote_url = self.remote_url[:-4]
                self.github_owner, self.github_repo_name = self.remote_url.split('/')[-2:]

            print(f"Attempting to connect to repo: owner='{self.github_owner}', name='{self.github_repo_name}'")

            # Get the GitHub repository object
            self.github_repo = self.g.get_user(self.github_owner).get_repo(self.github_repo_name)

        except git.GitError as e:
            print(f"Error initializing Git repository: {e}")
            self.repo = None
        except Exception as e:
            print(f"Error initializing GitHub client or finding repository: {e}")
            self.g = None

    def run_automerge_workflow(self):
        """
        This method executes the entire workflow on a repository with staged changes.
        It commits, pushes, creates a PR, and merges it.
        """
        if not self.repo or not self.g:
            print("Initialization failed. Cannot proceed.")
            return

        # Check for staged changes
        if not self.repo.index.diff(None) and not self.repo.untracked_files:
            print("No staged changes to commit. Aborting automerge workflow.")
            return

        # Define branch names and commit message
        current_branch = self.repo.active_branch.name
        new_branch_name = f'automerge-branch-{os.urandom(4).hex()}'
        commit_message = f"Auto: Update files via automerge"
        pr_title = f"Auto: Updates via automerge"
        pr_body = f"This PR contains automated updates."

        try:
            # Create a new branch
            new_branch = self.repo.create_head(new_branch_name)
            new_branch.checkout()

            # Commit the staged changes
            self.repo.index.commit(commit_message)
            print(f"Committed changes on branch '{new_branch_name}'.")

            # Push the new branch to the remote
            print(f"Pushing branch '{new_branch_name}' to remote '{self.remote_name}'...")
            self.repo.git.push(self.remote_name, new_branch_name, set_upstream=True)
            print("Push successful.")

            # Create the pull request
            print("Creating pull request...")
            pr = self.github_repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=new_branch_name,
                base=current_branch
            )
            print(f"Pull request created: {pr.html_url}")

            print("Automerge is enabled. Merging the PR...")
            pr.merge()
            print("Pull request merged successfully.")

            # Optional: delete the remote branch after merge
            #print("Deleting remote branch...")
            #self.github_repo.get_git_ref(f"heads/{new_branch_name}").delete()
            #print("Remote branch deleted.")

        except git.GitError as e:
            print(f"Git operation failed: {e}")
        except Exception as e:
            print(f"GitHub API operation failed: {e}")
        finally:
            # Always check out the original branch
            self.repo.git.checkout(current_branch)
            # Optional: delete the local branch
            self.repo.delete_head(new_branch_name)
            print(f"Switched back to branch '{current_branch}' and deleted local branch '{new_branch_name}'.")
