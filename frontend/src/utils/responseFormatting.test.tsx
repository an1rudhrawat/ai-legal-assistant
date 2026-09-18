import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FormattedResponse, speechText } from "./responseFormatting";

const response = "**What to do now**\n- Keep **your receipt**. [SOURCE:bnss-1]\n\nThis is general legal information, not legal advice. Consult a licensed advocate for your specific situation.";

describe("response formatting", () => {
  it("renders headings and bold emphasis while hiding internal citations and duplicate disclaimer", () => {
    render(<FormattedResponse text={response} />);
    expect(screen.getByRole("heading", { name: "What to do now" })).toBeInTheDocument();
    expect(screen.getByText("your receipt").tagName).toBe("STRONG");
    expect(screen.queryByText(/SOURCE:bnss/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^This is general legal information/)).not.toBeInTheDocument();
  });

  it("removes formatting symbols, citations, and disclaimer before speech", () => {
    expect(speechText(response)).toBe("What to do now Keep your receipt.");
  });
});
