import aiohttp
import asyncio
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import logging
from urllib.parse import urljoin, urlparse
import re

logger = logging.getLogger(__name__)

class InternetSearch:
    """Real-time internet access for code context and documentation"""
    
    def __init__(self, max_results: int = 5, timeout: int = 10):
        self.max_results = max_results
        self.timeout = timeout
        self.session = None
        self.search_engines = [
            "https://www.google.com/search?q=",
            "https://duckduckgo.com/?q=",
        ]
    
    async def search(self, query: str, language_hint: str = "") -> List[Dict[str, Any]]:
        """Search the internet for relevant documentation and code examples"""
        try:
            async with aiohttp.ClientSession() as session:
                results = []
                
                # Search for documentation
                doc_results = await self._search_documentation(session, query, language_hint)
                results.extend(doc_results)
                
                # Search for Stack Overflow
                so_results = await self._search_stackoverflow(session, query)
                results.extend(so_results)
                
                # Search for GitHub
                gh_results = await self._search_github(session, query, language_hint)
                results.extend(gh_results)
                
                return results[:self.max_results]
        
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    
    async def _search_documentation(self, session, query: str, language_hint: str) -> List[Dict[str, Any]]:
        """Search official documentation"""
        doc_urls = {
            "python": "https://docs.python.org/3/search.html?q=",
            "javascript": "https://developer.mozilla.org/en-US/search?q=",
            "java": "https://docs.oracle.com/javase/16/docs/api/search.html?q=",
            "cpp": "https://en.cppreference.com/mwiki/index.php?search=",
            "rust": "https://doc.rust-lang.org/std/?search=",
            "go": "https://golang.org/search?q=",
        }
        
        results = []
        base_url = doc_urls.get(language_hint.lower())
        
        if base_url:
            try:
                url = base_url + query.replace(" ", "+")
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract search results
                        for item in soup.find_all(['a', 'div'], class_=re.compile('result|link', re.I))[:3]:
                            if item.get('href'):
                                results.append({
                                    "source": "documentation",
                                    "title": item.get_text()[:100],
                                    "url": urljoin(base_url, item.get('href')),
                                    "snippet": item.get_text()[:200]
                                })
            except Exception as e:
                logger.warning(f"Documentation search error: {str(e)}")
        
        return results
    
    async def _search_stackoverflow(self, session, query: str) -> List[Dict[str, Any]]:
        """Search Stack Overflow for Q&A"""
        results = []
        try:
            url = f"https://api.stackexchange.com/2.3/search?order=desc&sort=relevance&intitle={query}&site=stackoverflow&pagesize=5"
            
            async with session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for item in data.get('items', [])[:3]:
                        results.append({
                            "source": "stackoverflow",
                            "title": item.get('title', '')[:100],
                            "url": item.get('link', ''),
                            "snippet": f"Score: {item.get('score', 0)} | Answers: {item.get('answer_count', 0)}",
                            "score": item.get('score', 0)
                        })
        except Exception as e:
            logger.warning(f"Stack Overflow search error: {str(e)}")
        
        return results
    
    async def _search_github(self, session, query: str, language_hint: str) -> List[Dict[str, Any]]:
        """Search GitHub for code examples"""
        results = []
        try:
            lang_param = f"language:{language_hint}" if language_hint else ""
            url = f"https://api.github.com/search/code?q={query}+{lang_param}&per_page=5"
            
            headers = {"Accept": "application/vnd.github.v3+json"}
            async with session.get(url, headers=headers, timeout=self.timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for item in data.get('items', [])[:3]:
                        results.append({
                            "source": "github",
                            "title": f"{item.get('repository', {}).get('name', '')} - {item.get('name', '')}",
                            "url": item.get('html_url', ''),
                            "snippet": item.get('path', '')[:100],
                            "stars": item.get('repository', {}).get('stargazers_count', 0)
                        })
        except Exception as e:
            logger.warning(f"GitHub search error: {str(e)}")
        
        return results
    
    async def fetch_documentation(self, url: str) -> str:
        """Fetch and parse documentation from URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = '\n'.join(chunk for chunk in chunks if chunk)
                        
                        return text[:2000]  # Limit to 2000 chars
        except Exception as e:
            logger.error(f"Documentation fetch error: {str(e)}")
        
        return ""
    
    async def search_with_caching(self, query: str, language: str = "") -> List[Dict[str, Any]]:
        """Search with simple caching"""
        cache_key = f"{query}_{language}".lower()
        
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        results = await self.search(query, language)
        self._cache[cache_key] = results
        
        # Limit cache size
        if len(self._cache) > 100:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        return results

class APIDocumentation:
    """Manage API documentation for code completion"""
    
    def __init__(self):
        self.docs_cache = {}
        self.internet_search = InternetSearch()
    
    async def get_api_docs(self, language: str, library: str) -> Dict[str, Any]:
        """Get API documentation for a library"""
        cache_key = f"{language}_{library}"
        
        if cache_key in self.docs_cache:
            return self.docs_cache[cache_key]
        
        # Fetch from internet
        query = f"{library} {language} api documentation"
        results = await self.internet_search.search(query, language)
        
        self.docs_cache[cache_key] = {
            "library": library,
            "language": language,
            "docs": results
        }
        
        return self.docs_cache[cache_key]
    
    async def complete_code(self, partial_code: str, language: str) -> List[str]:
        """Suggest code completions based on internet docs"""
        # Parse partial code for function/class names
        import re
        match = re.search(r'(\w+)\.$', partial_code)
        
        if match:
            base = match.group(1)
            # Search for available methods/properties
            results = await self.internet_search.search(f"{base} methods api {language}", language)
            
            suggestions = []
            for result in results:
                suggestions.append(result.get('title', ''))
            
            return suggestions
        
        return []
