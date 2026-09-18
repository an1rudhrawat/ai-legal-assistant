import { Fragment, ReactNode } from "react";

const DISCLAIMER = "This is general legal information, not legal advice. Consult a licensed advocate for your specific situation.";
const INTERNAL_CITATION = /\s*\[(?:SOURCE:)?[A-Za-z0-9_-]+\]/g;

/** Remove transport-only material before rendering or speaking a model answer. */
export function cleanResponse(text: string): string {
  return text
    .replace(INTERNAL_CITATION, "")
    .replace(DISCLAIMER, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function speechText(text: string): string {
  return cleanResponse(text)
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/^#{1,6}\s*/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/\s+/g, " ")
    .trim();
}

function inlineMarkdown(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|__[^_]+__)/g).filter(Boolean).map((part, index) => {
    const bold = part.match(/^\*\*(.+)\*\*$/) || part.match(/^__(.+)__$/);
    return bold ? <strong key={index}>{bold[1]}</strong> : <Fragment key={index}>{part}</Fragment>;
  });
}

export function FormattedResponse({ text }: { text: string }) {
  const blocks = cleanResponse(text).split(/\n\s*\n/).filter(Boolean);
  return <div className="response-body">
    {blocks.map((block, index) => {
      const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
      const heading = lines[0]?.match(/^#{1,6}\s+(.+)/) || lines[0]?.match(/^\*\*(.+)\*\*$/);
      const content = heading ? lines.slice(1) : lines;
      const listItems = content.filter((line) => /^[-*+]\s+/.test(line));
      const renderedContent = listItems.length === content.length && content.length
        ? <ul>{listItems.map((line, itemIndex) => <li key={itemIndex}>{inlineMarkdown(line.replace(/^[-*+]\s+/, ""))}</li>)}</ul>
        : content.length ? <p>{content.map((line, lineIndex) => <Fragment key={lineIndex}>{lineIndex > 0 && <br />}{inlineMarkdown(line)}</Fragment>)}</p> : null;
      if (heading) {
        return <Fragment key={index}><h3>{heading[1]}</h3>{renderedContent}</Fragment>;
      }
      return <Fragment key={index}>{renderedContent}</Fragment>;
    })}
  </div>;
}
