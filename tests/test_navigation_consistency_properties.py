"""Property-based tests for autonomous navigation consistency (Property 1)."""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
from unittest.mock import AsyncMock, MagicMock
import asyncio

from sovereign_career_architect.browser.navigation import JobPortalNavigator, create_job_portal_navigator


# Strategies for generating test data
portal_strategy = st.sampled_from(["linkedin", "ycombinator", "angellist"])

job_query_strategy = st.text(min_size=3, max_size=50).filter(
    lambda x: len(x.strip()) > 2 and x.isascii()
)

location_strategy = st.one_of(
    st.none(),
    st.sampled_from([
        "San Francisco, CA", "New York, NY", "Remote", "Seattle, WA", 
        "Austin, TX", "Boston, MA", "Los Angeles, CA"
    ])
)

job_filters_strategy = st.one_of(
    st.none(),
    st.fixed_dictionaries({
        "job_type": st.sampled_from(["full-time", "part-time", "contract", "internship"]),
        "experience_level": st.sampled_from(["entry", "mid", "senior", "executive"]),
        "remote": st.booleans()
    })
)

job_index_strategy = st.integers(min_value=0, max_value=10)


class TestNavigationConsistencyProperties:
    """Property-based tests for navigation consistency and reliability."""
    
    @given(portal=portal_strategy)
    @settings(max_examples=15, deadline=3000)
    @pytest.mark.asyncio
    async def test_portal_navigation_consistency(self, portal):
        """
        Property 1a: Portal navigation should be consistent and idempotent.
        
        Navigating to the same portal multiple times should always succeed
        and maintain consistent state.
        """
        navigator = create_job_portal_navigator()
        
        # Navigate to portal multiple times
        results = []
        for _ in range(3):
            result = await navigator.navigate_to_portal(portal)
            results.append(result)
            
            # Verify state consistency
            assert navigator.current_portal == portal
        
        # All navigation attempts should succeed
        assert all(results), f"Inconsistent navigation results: {results}"
        
        # Navigation history should reflect all attempts
        portal_navigations = [
            h for h in navigator.get_navigation_history() 
            if h["action"] == "navigate_to_portal" and h["portal"] == portal
        ]
        assert len(portal_navigations) == 3
    
    @given(
        portal=portal_strategy,
        query=job_query_strategy,
        location=location_strategy
    )
    @settings(max_examples=20, deadline=5000)
    @pytest.mark.asyncio
    async def test_job_search_consistency(self, portal, query, location):
        """
        Property 1b: Job search should return consistent results for identical queries.
        
        Performing the same search multiple times should return similar results
        in terms of structure and basic properties.
        """
        navigator = create_job_portal_navigator()
        
        # Navigate to portal first
        await navigator.navigate_to_portal(portal)
        
        # Perform same search multiple times
        search_results = []
        for _ in range(2):
            jobs = await navigator.search_jobs(query, location)
            search_results.append(jobs)
        
        # Verify consistency
        assert len(search_results) == 2
        
        if search_results[0]:  # If jobs were found
            # Results should have similar structure
            for result_set in search_results:
                assert isinstance(result_set, list)
                
                for job in result_set:
                    # Each job should have required fields
                    required_fields = ["title", "company", "portal", "url"]
                    for field in required_fields:
                        assert field in job, f"Missing field '{field}' in job"
                    
                    # Portal should match
                    assert job["portal"] == portal
                    
                    # URL should be valid format
                    assert isinstance(job["url"], str)
                    assert len(job["url"]) > 0
            
            # Results should be reasonably consistent in count
            counts = [len(result) for result in search_results]
            if max(counts) > 0:
                # Allow some variance but should be generally consistent
                variance = max(counts) - min(counts)
                assert variance <= max(2, max(counts) * 0.5), \
                    f"Inconsistent result counts: {counts}"
    
    @given(
        portal=portal_strategy,
        query=job_query_strategy,
        filters=job_filters_strategy
    )
    @settings(max_examples=15, deadline=4000)
    @pytest.mark.asyncio
    async def test_search_with_filters_consistency(self, portal, query, filters):
        """
        Property 1c: Search filters should consistently affect results.
        
        Applying the same filters should produce consistent filtering behavior.
        """
        navigator = create_job_portal_navigator()
        await navigator.navigate_to_portal(portal)
        
        # Search with filters
        jobs_with_filters = await navigator.search_jobs(query, filters=filters)
        
        # Search without filters for comparison
        jobs_without_filters = await navigator.search_jobs(query)
        
        # Verify filter consistency
        assert isinstance(jobs_with_filters, list)
        assert isinstance(jobs_without_filters, list)
        
        # Both searches should return valid job structures
        for job_list in [jobs_with_filters, jobs_without_filters]:
            for job in job_list:
                assert isinstance(job, dict)
                assert "title" in job
                assert "portal" in job
                assert job["portal"] == portal
    
    @given(
        portal=portal_strategy,
        query=job_query_strategy
    )
    @settings(max_examples=10, deadline=4000)
    @pytest.mark.asyncio
    async def test_job_navigation_consistency(self, portal, query):
        """
        Property 1d: Job navigation should be consistent and maintain state.
        
        Navigating to jobs should maintain consistent behavior and state tracking.
        """
        navigator = create_job_portal_navigator()
        
        # Set up search results
        await navigator.navigate_to_portal(portal)
        jobs = await navigator.search_jobs(query)
        
        if jobs:
            # Ensure jobs have valid URLs for navigation
            for i, job in enumerate(jobs):
                if job.get("url") == "#" or not job.get("url"):
                    jobs[i]["url"] = f"https://{portal}.com/job/{i+1}"
            
            navigator.current_search_results = jobs
            
            # Test navigation to first job
            if len(jobs) > 0:
                result = await navigator.navigate_to_job(0)
                assert result is True
                
                # Navigation history should be updated
                job_navigations = [
                    h for h in navigator.get_navigation_history()
                    if h["action"] == "navigate_to_job"
                ]
                assert len(job_navigations) >= 1
                
                # Test navigation to same job again (should be consistent)
                result2 = await navigator.navigate_to_job(0)
                assert result2 is True
    
    @given(portal=portal_strategy)
    @settings(max_examples=10, deadline=3000)
    @pytest.mark.asyncio
    async def test_navigation_state_consistency(self, portal):
        """
        Property 1e: Navigation state should remain consistent throughout operations.
        
        The navigator's internal state should be consistent and predictable.
        """
        navigator = create_job_portal_navigator()
        
        # Initial state should be clean
        assert navigator.current_portal is None
        assert navigator.current_search_results == []
        assert navigator.navigation_history == []
        
        # After portal navigation
        await navigator.navigate_to_portal(portal)
        assert navigator.current_portal == portal
        assert len(navigator.navigation_history) == 1
        
        # After job search
        jobs = await navigator.search_jobs("Engineer")
        assert navigator.current_search_results == jobs
        assert len(navigator.navigation_history) == 2
        
        # State accessors should return consistent copies
        history1 = navigator.get_navigation_history()
        history2 = navigator.get_navigation_history()
        assert history1 == history2
        assert history1 is not history2  # Should be different objects (copies)
        
        results1 = navigator.get_current_search_results()
        results2 = navigator.get_current_search_results()
        assert results1 == results2
        assert results1 is not results2  # Should be different objects (copies)
    
    @given(
        portal=portal_strategy,
        invalid_index=st.integers(min_value=-10, max_value=-1)
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_invalid_navigation_consistency(self, portal, invalid_index):
        """
        Property 1f: Invalid navigation attempts should consistently fail.
        
        Invalid operations should always fail in a predictable manner.
        """
        navigator = create_job_portal_navigator()
        
        # Set up some search results
        await navigator.navigate_to_portal(portal)
        jobs = await navigator.search_jobs("Engineer")
        
        # Test invalid job index navigation
        result = await navigator.navigate_to_job(invalid_index)
        assert result is False
        
        # Test navigation with out-of-range positive index
        large_index = len(jobs) + 10
        result = await navigator.navigate_to_job(large_index)
        assert result is False
        
        # Test navigation to unknown portal
        result = await navigator.navigate_to_portal("unknown_portal_xyz")
        assert result is False
        
        # Current portal should remain unchanged after failed navigation
        assert navigator.current_portal == portal


class NavigationStateMachine(RuleBasedStateMachine):
    """Stateful testing for navigation system consistency."""
    
    def __init__(self):
        super().__init__()
        self.navigator = create_job_portal_navigator()
        self.valid_portals = ["linkedin", "ycombinator", "angellist"]
        self.performed_searches = []
    
    @initialize()
    def setup(self):
        """Initialize the state machine."""
        self.navigator = create_job_portal_navigator()
        self.performed_searches = []
    
    @rule(portal=st.sampled_from(["linkedin", "ycombinator", "angellist"]))
    def navigate_to_portal(self, portal):
        """Rule: Navigate to a job portal."""
        async def _navigate():
            return await self.navigator.navigate_to_portal(portal)
        
        # Run async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_navigate())
        finally:
            loop.close()
        
        # Navigation to valid portal should always succeed
        assert result is True
        assert self.navigator.current_portal == portal
    
    @rule(
        query=st.text(min_size=3, max_size=30).filter(lambda x: x.strip() and x.isascii()),
        location=st.one_of(st.none(), st.sampled_from(["Remote", "San Francisco", "New York"]))
    )
    def search_jobs(self, query, location):
        """Rule: Search for jobs on current portal."""
        # Only search if we have a current portal
        assume(self.navigator.current_portal is not None)
        
        async def _search():
            return await self.navigator.search_jobs(query, location)
        
        # Run async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            jobs = loop.run_until_complete(_search())
        finally:
            loop.close()
        
        # Search should return a list
        assert isinstance(jobs, list)
        
        # Track this search
        self.performed_searches.append({
            "query": query,
            "location": location,
            "portal": self.navigator.current_portal,
            "result_count": len(jobs)
        })
        
        # Verify job structure
        for job in jobs:
            assert isinstance(job, dict)
            assert "title" in job
            assert "portal" in job
            assert job["portal"] == self.navigator.current_portal
    
    @rule(job_index=st.integers(min_value=0, max_value=5))
    def navigate_to_job(self, job_index):
        """Rule: Navigate to a specific job."""
        # Only navigate if we have search results
        assume(len(self.navigator.current_search_results) > 0)
        
        async def _navigate_job():
            return await self.navigator.navigate_to_job(job_index)
        
        # Run async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_navigate_job())
        finally:
            loop.close()
        
        # Result should depend on index validity
        if job_index < len(self.navigator.current_search_results):
            # Valid index should succeed (in mock mode)
            assert result is True
        else:
            # Invalid index should fail
            assert result is False
    
    @invariant()
    def navigation_state_consistency(self):
        """Invariant: Navigation state should always be consistent."""
        # Current portal should be valid or None
        if self.navigator.current_portal is not None:
            assert self.navigator.current_portal in self.valid_portals
        
        # Search results should be a list
        assert isinstance(self.navigator.current_search_results, list)
        
        # Navigation history should be a list
        assert isinstance(self.navigator.navigation_history, list)
        
        # All jobs in search results should have required fields
        for job in self.navigator.current_search_results:
            assert isinstance(job, dict)
            assert "title" in job
            assert "portal" in job
            if self.navigator.current_portal:
                assert job["portal"] == self.navigator.current_portal
    
    @invariant()
    def history_consistency(self):
        """Invariant: Navigation history should be consistent and ordered."""
        history = self.navigator.navigation_history
        
        # History entries should have required fields
        for entry in history:
            assert isinstance(entry, dict)
            assert "action" in entry
            assert "timestamp" in entry
            
            # Timestamps should be in order (non-decreasing)
            if len(history) > 1:
                for i in range(1, len(history)):
                    assert history[i]["timestamp"] >= history[i-1]["timestamp"]


# Stateful test
TestNavigationStateMachine = NavigationStateMachine.TestCase


@pytest.mark.property
class TestNavigationIntegrationProperties:
    """Integration property tests for navigation system."""
    
    @given(
        portals=st.lists(portal_strategy, min_size=1, max_size=3, unique=True),
        queries=st.lists(job_query_strategy, min_size=1, max_size=3)
    )
    @settings(max_examples=8, deadline=10000)
    @pytest.mark.asyncio
    async def test_multi_portal_navigation_consistency(self, portals, queries):
        """
        Property 1g: Multi-portal navigation should maintain consistency.
        
        Navigating across multiple portals should maintain consistent behavior
        and state management.
        """
        navigator = create_job_portal_navigator()
        
        portal_results = {}
        
        for portal in portals:
            # Navigate to portal
            result = await navigator.navigate_to_portal(portal)
            assert result is True
            assert navigator.current_portal == portal
            
            # Perform searches on this portal
            portal_jobs = []
            for query in queries:
                jobs = await navigator.search_jobs(query)
                portal_jobs.extend(jobs)
                
                # Verify all jobs belong to current portal
                for job in jobs:
                    assert job["portal"] == portal
            
            portal_results[portal] = portal_jobs
        
        # Verify navigation history consistency
        history = navigator.get_navigation_history()
        portal_navigations = [h for h in history if h["action"] == "navigate_to_portal"]
        
        # Should have one navigation per portal
        assert len(portal_navigations) == len(portals)
        
        # Portal names in history should match our navigation order
        navigated_portals = [h["portal"] for h in portal_navigations]
        assert navigated_portals == portals
    
    @given(
        portal=portal_strategy,
        query=job_query_strategy
    )
    @settings(max_examples=10, deadline=5000)
    @pytest.mark.asyncio
    async def test_navigation_error_recovery_consistency(self, portal, query):
        """
        Property 1h: Navigation should consistently recover from errors.
        
        Error conditions should be handled consistently without corrupting state.
        """
        navigator = create_job_portal_navigator()
        
        # Successful navigation
        result = await navigator.navigate_to_portal(portal)
        assert result is True
        
        # Attempt invalid operations
        invalid_result = await navigator.navigate_to_portal("invalid_portal")
        assert invalid_result is False
        
        # State should remain consistent after error
        assert navigator.current_portal == portal
        
        # Should still be able to perform valid operations
        jobs = await navigator.search_jobs(query)
        assert isinstance(jobs, list)
        
        # Invalid job navigation
        invalid_job_result = await navigator.navigate_to_job(-1)
        assert invalid_job_result is False
        
        # State should still be consistent
        assert navigator.current_portal == portal
        assert navigator.current_search_results == jobs