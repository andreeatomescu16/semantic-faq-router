from app.services.router import DomainRouter


def test_domain_router_in_scope_query() -> None:
    router = DomainRouter()
    result = router.route_domain("How do I change my password?")
    assert result.in_domain is True
    assert result.category == "security"


def test_domain_router_out_of_scope_query() -> None:
    router = DomainRouter()
    result = router.route_domain("What is the weather in Tokyo?")
    assert result.in_domain is False
    assert result.category == "N/A"
