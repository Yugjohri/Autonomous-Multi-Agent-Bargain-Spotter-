"""Test doubles shared by several test modules. No network access anywhere."""

from agents.normalize import is_store_url


class FakeResponse:
    def __init__(self, status=200, location=None, text=""):
        self.status_code = status
        self.headers = {"Location": location} if location else {}
        self.text = text

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHttp:
    """
    Answers redirect requests from a {url: location} map. Records every request and
    fails the test if anything ever asks for a store product page.
    """

    offline = False

    def __init__(self, redirects=None, disallowed_hosts=(), head_unsupported=(), pages=None):
        self.redirects = dict(redirects or {})
        self.disallowed_hosts = set(disallowed_hosts)
        self.head_unsupported = set(head_unsupported)
        self.pages = dict(pages or {})
        self.requests = []

    def allowed(self, url):
        from agents.normalize import host_of

        return host_of(url) not in self.disallowed_hosts

    def request(self, method, url, **kwargs):
        assert not is_store_url(url) or url.startswith("https://dl.flipkart.com/s/"), f"requested store page {url}"
        self.requests.append((method, url))
        if method == "HEAD" and url in self.head_unsupported:
            return FakeResponse(404)
        if url in self.redirects:
            return FakeResponse(301, self.redirects[url])
        if url in self.pages:
            return FakeResponse(200, text=self.pages[url])
        return FakeResponse(200)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def head(self, url, **kwargs):
        return self.request("HEAD", url, **kwargs)
