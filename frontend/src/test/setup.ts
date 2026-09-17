import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Keep each component test isolated when the test runner reuses the DOM.
afterEach(() => cleanup());
