from app.llm.llama_client import LlamaClient
from app.safety.input_guard import InputGuard
from app.safety.output_guard import OutputGuard


class SafetyGateway:
    def __init__(self):
        print("Initializing Neural Feature Safety Gateway...")

        self.input_guard = InputGuard()
        self.output_guard = OutputGuard(
            safety_model=self.input_guard.safety_model
        )
        self.llama = LlamaClient()

        self.max_recovery_attempts = 2

        print("Neural Feature Safety Gateway ready.")

    def _generate_safe_alternative(self, original_prompt):
        recovery_prompt = (
            "Provide a safe and helpful response to the user's request. "
            "Do not provide instructions that could enable harm, illegal "
            "activity, violence, exploitation, or unauthorized access. "
            "If the request is unsafe, briefly explain that you cannot "
            "provide the requested harmful instructions and redirect the "
            "user toward a legitimate and safe alternative.\n\n"
            f"User request:\n{original_prompt}"
        )

        return self.llama.generate(
            recovery_prompt,
            max_new_tokens=200,
            temperature=0.2
        )

    def _generate_normal_response(self, prompt):
        return self.llama.generate(
            prompt,
            max_new_tokens=200,
            temperature=0.2
        )

    def _validate_output(self, response):
        return self.output_guard.analyze(response)

    def _build_fallback(self):
        fallback_candidates = [
            "I can't provide that request. I can help with a safe and lawful alternative.",
            "I can't provide instructions that could cause harm. I can help with a safe alternative.",
            "I can't assist with that request. I can help with information for a legitimate and safe purpose."
        ]

        for response in fallback_candidates:
            output_result = self._validate_output(response)

            if output_result["allowed"]:
                return response, output_result

        return None, output_result

    def _result(
        self,
        response,
        input_result,
        output_result,
        path,
        recovery_attempts
    ):
        return {
            "response": response,
            "input_safety": input_result,
            "output_safety": output_result,
            "path": path,
            "recovery_attempts": recovery_attempts
        }

    def chat(self, prompt):
        prompt = prompt.strip()

        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        input_result = self.input_guard.analyze(prompt)

        if input_result["decision"] == "BLOCK":
            response_result = self._generate_safe_alternative(prompt)
            response = response_result["response"]
            output_result = self._validate_output(response)

            if output_result["allowed"]:
                return self._result(
                    response,
                    input_result,
                    output_result,
                    "blocked_input_safe_alternative",
                    0
                )

            for attempt in range(1, self.max_recovery_attempts + 1):
                recovery_result = self._generate_safe_alternative(prompt)
                response = recovery_result["response"]
                output_result = self._validate_output(response)

                if output_result["allowed"]:
                    return self._result(
                        response,
                        input_result,
                        output_result,
                        "blocked_input_recovery",
                        attempt
                    )

            fallback_response, fallback_result = self._build_fallback()

            if fallback_response is not None:
                return self._result(
                    fallback_response,
                    input_result,
                    fallback_result,
                    "deterministic_fallback",
                    self.max_recovery_attempts
                )

            raise RuntimeError(
                "No safe response passed the output safety check."
            )

        generation_result = self._generate_normal_response(prompt)
        response = generation_result["response"]
        output_result = self._validate_output(response)

        if output_result["allowed"]:
            return self._result(
                response,
                input_result,
                output_result,
                "normal_generation",
                0
            )

        for attempt in range(1, self.max_recovery_attempts + 1):
            recovery_result = self._generate_safe_alternative(prompt)
            response = recovery_result["response"]
            output_result = self._validate_output(response)

            if output_result["allowed"]:
                return self._result(
                    response,
                    input_result,
                    output_result,
                    "output_recovery",
                    attempt
                )

        fallback_response, fallback_result = self._build_fallback()

        if fallback_response is not None:
            return self._result(
                fallback_response,
                input_result,
                fallback_result,
                "deterministic_fallback",
                self.max_recovery_attempts
            )

        raise RuntimeError(
            "No safe response passed the output safety check."
        )