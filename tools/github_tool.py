import requests
# pyrefly: ignore [missing-import]
from crewai.tools import tool

@tool("analyze_github_profile")
def analyze_github_profile(github_username: str) -> str:
    """Useful to fetch and analyze a candidate's real public GitHub repositories and programming languages.
    Requires the exact github username. Do not pass full URLs, only the username."""
    if not github_username or not str(github_username).strip():
        return "No GitHub username provided."

    clean_user = str(github_username).strip().rstrip('/')
    # Handle full URLs if passed by LLM (e.g. https://github.com/username)
    if 'github.com/' in clean_user:
        clean_user = clean_user.split('github.com/')[-1].split('/')[0]
    elif '/' in clean_user:
        clean_user = clean_user.split('/')[-1]
    if clean_user.startswith('@'):
        clean_user = clean_user[1:]

    try:
        url = f"https://api.github.com/users/{clean_user}/repos?sort=updated&per_page=5"
        headers = {'User-Agent': 'SmartHire-Recruitment-AI'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 404:
            return f"GitHub profile '{clean_user}' not found. Candidate may not have public repositories."
        if response.status_code != 200:
            return f"Failed to fetch GitHub profile for '{clean_user}'. Status Code: {response.status_code}. Rate limit may be exceeded."
            
        repos = response.json()
        if not repos or not isinstance(repos, list):
            return f"GitHub profile '{clean_user}' has no public repositories."
            
        analysis = f"GitHub Portfolio Analysis for '{clean_user}':\n"
        for repo in repos:
            name = repo.get("name", "Unknown")
            language = repo.get("language", "Unknown")
            description = repo.get("description", "No description")
            stars = repo.get("stargazers_count", 0)
            
            analysis += f"- Repo: {name} | Language: {language} | Stars: {stars}\n"
            if description:
                analysis += f"  Desc: {description}\n"
            
        return analysis
    except Exception as e:
        return f"Error connecting to GitHub API: {str(e)}"

def get_github_tool():
    return analyze_github_profile
