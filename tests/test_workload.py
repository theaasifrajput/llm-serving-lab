from llm_serving_lab.workload.generator import (
    create_fixed_workload,
)


def test_fixed_workload():

    workload = create_fixed_workload(
        name="basic",
        prompt="Hello world",
        requests=10,
        max_tokens=64,
    )

    assert workload.name == "basic"
    assert workload.size == 10

    assert all(
        request.prompt == "Hello world"
        for request in workload.requests
    )

    assert all(
        request.max_tokens == 64
        for request in workload.requests
    )