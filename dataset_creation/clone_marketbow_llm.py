import os 
import subprocess 
from multiprocessing import Pool 
from github import Github 

ORG = "huggingface" 
MIRROR_DIRECTORY = "marketbow_llm" 
TOK_K = 15 


def get_repos(username, access_token=None, include_fork=False):
    """ """
    g = Github(access_token)
    user = g.get_user(username) 
    
    results = [] 
    for repo in user.get_repos():
        if repo.fork is False: 
            results.append((repo.name, repo.stargazers_count))
        else:
            if include_fork is True: 
                results.append((repo.name, repo.stargazers_count))
    print(results)
    return results 


def sort_repos_by_stars(repos):
    return sorted(repos, key=lambda x: x[1], reverse=True)


def mirror_repository(repository):
    """Locally clones a repository."""
    repository_url = f"https://github.com/{ORG}/{repository}.git" 
    repository_path = os.path.join(MIRROR_DIRECTORY, repository)
    
    # Clone the repository 
    subprocess.run(["git", "clone", repository_url, repository_path])
    
    
def mirror_repositories():
    pass 