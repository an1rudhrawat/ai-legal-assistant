import { Fragment, ReactNode } from "react";

const DISCLAIMER = "This is general legal information, not legal advice. Consult a licensed advocate for your specific situation.";
const INTERNAL_CITATION = /\s*\[(?:SOURCE:)?[A-Za-z0-9_-]+\]/gi;

/** Remove transport-only material and common PDF text-encoding artifacts. */
export function cleanResponse(text: string): string {
  return text
    .replace(/â€”/g, "—")
    .replace(/â€“/g, "–")
    .replace(/â€™/g, "’")
    .replace(/â€œ/g, "“")
    .replace(/â€�/g, "”")
    .replace(/â€¦/g, "…")
    .replace(/(?:^|\n)[\t ]*(?:source identifier|source id|citation)\s*:[^\n]*(?=\n|$)/gim, "")
    .replace(INTERNAL_CITATION, "")
    .replace(DISCLAIMER, "")
    .replace(/^\s*[-–—]\s*$/gm, "")
    .replace(/\n[\t ]*\n(?:[\t ]*\n)+/g, "\n\n")
    .trim();
}

export function speechText(text: string): string {
  return cleanResponse(text)
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/^#{1,6}\s*/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+[.)]\s+/gm, "")
    .replace(/[`*_]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function inlineMarkdown(text: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|__[^_]+__|\*[^*\n]+\*|_[^_\n]+_|`[^`]+`)/g).filter(Boolean);
  return parts.map((part, index) => {
    const bold = part.match(/^\*\*(.+)\*\*$/) || part.match(/^__(.+)__$/);
    const italic = part.match(/^\*(.+)\*$/) || part.match(/^_(.+)_$/);
    const code = part.match(/^`(.+)`$/);
    if (bold) return <strong key={index}>{bold[1]}</strong>;
    if (italic) return <em key={index}>{italic[1]}</em>;
    if (code) return <code key={index}>{code[1]}</code>;
    return <Fragment key={index}>{part}</Fragment>;
  });
}

function renderBlock(block: string, blockIndex: number): ReactNode[] {
  const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
  const nodes: ReactNode[] = [];
  let paragraph: string[] = [];
  let index = 0;
  const flushParagraph = () => {
    if (!paragraph.length) return;
    nodes.push(<p key={`${blockIndex}-p-${index++}`}>{paragraph.map((line, lineIndex) => <Fragment key={lineIndex}>{lineIndex > 0 && <br />}{inlineMarkdown(line)}</Fragment>)}</p>);
    paragraph = [];
  };

  while (index < lines.length) {
    const line = lines[index];
    if (/^\s*(?:[-*_]\s*){3,}$/.test(line)) {
      flushParagraph();
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/) || line.match(/^\*\*([^*]+)\*\*:?\s*$/);
    if (heading) {
      flushParagraph();
      const level = heading[1] ? Math.min(heading[1].length + 1, 4) : 3;
      const title = heading[2] ?? heading[1];
      const Heading = `h${level}` as "h2" | "h3" | "h4";
      nodes.push(<Heading key={`${blockIndex}-h-${index++}`}>{title.replace(/:$/, "")}</Heading>);
      continue;
    }

    if (/^[-*+]\s+/.test(line)) {
      flushParagraph();
      const items: string[] = [];
      while (index < lines.length && /^[-*+]\s+/.test(lines[index])) items.push(lines[index++].replace(/^[-*+]\s+/, ""));
      nodes.push(<ul key={`${blockIndex}-ul-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}>{inlineMarkdown(item)}</li>)}</ul>);
      continue;
    }

    if (/^\d+[.)]\s+/.test(line)) {
      flushParagraph();
      const items: string[] = [];
      while (index < lines.length && /^\d+[.)]\s+/.test(lines[index])) items.push(lines[index++].replace(/^\d+[.)]\s+/, ""));
      nodes.push(<ol key={`${blockIndex}-ol-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}>{inlineMarkdown(item)}</li>)}</ol>);
      continue;
    }

    if (/^>\s?/.test(line)) {
      flushParagraph();
      const quote: string[] = [];
      while (index < lines.length && /^>\s?/.test(lines[index])) quote.push(lines[index++].replace(/^>\s?/, ""));
      nodes.push(<blockquote key={`${blockIndex}-quote-${index}`}>{quote.map((part, quoteIndex) => <Fragment key={quoteIndex}>{quoteIndex > 0 && <br />}{inlineMarkdown(part)}</Fragment>)}</blockquote>);
      continue;
    }

    paragraph.push(line);
    index += 1;
  }
  flushParagraph();
  return nodes;
}

export function FormattedResponse({ text }: { text: string }) {
  const blocks = cleanResponse(text).split(/\n\s*\n/).filter(Boolean);
  return <div className="response-body">{blocks.flatMap((block, index) => renderBlock(block, index))}</div>;
}
