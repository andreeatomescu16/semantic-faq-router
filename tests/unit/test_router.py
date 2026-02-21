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


def test_domain_router_password_reset_synonyms() -> None:
    router = DomainRouter()
    result = router.route_domain("forgot pass, how reset?")
    assert result.in_domain is True
    assert result.category == "security"


def test_domain_router_export_data_phrase() -> None:
    router = DomainRouter()
    result = router.route_domain("How can I export all my data?")
    assert result.in_domain is True
    assert result.category == "privacy"


def test_domain_router_subscription_cancel_semantics() -> None:
    router = DomainRouter()
    result = router.route_domain("stop renewal but keep access till end")
    assert result.in_domain is True
    assert result.category == "subscription"


def test_domain_router_developer_api_key_query() -> None:
    router = DomainRouter()
    result = router.route_domain("where do i create developer api key")
    assert result.in_domain is True
    assert result.category == "developer"
