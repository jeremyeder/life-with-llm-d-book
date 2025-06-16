import React, {useState, useRef, useEffect} from 'react';
import clsx from 'clsx';
import {useThemeConfig, usePrismTheme} from '@docusaurus/theme-common';
import {
  parseCodeBlockTitle,
  parseLanguage,
  parseLines,
  containsLineNumbers,
  useCodeWordWrap,
} from '@docusaurus/theme-common/internal';
import {Highlight, type Language} from 'prism-react-renderer';
import Line from '@theme/CodeBlock/Line';
import CopyButton from '@theme/CodeBlock/CopyButton';
import WordWrapButton from '@theme/CodeBlock/WordWrapButton';
import Container from '@theme/CodeBlock/Container';
import type {Props} from '@theme/CodeBlock/Content/String';

import styles from './styles.module.css';

// Prism languages are always lowercase
// We want to fail-safe and allow both "php" and "PHP"
// See https://github.com/facebook/docusaurus/issues/9012
function normalizeLanguage(language: string | undefined): string | undefined {
  return language?.toLowerCase();
}

// Custom expand button component
function ExpandButton({isExpanded, onClick, isShort}: {
  isExpanded: boolean;
  onClick: () => void;
  isShort: boolean;
}) {
  if (isShort) return null;
  
  return (
    <button
      className="code-expand-button"
      onClick={onClick}
      aria-label={isExpanded ? 'Collapse code' : 'Expand code'}
      type="button"
    >
      {isExpanded ? '▲ Collapse' : '▼ Expand'}
    </button>
  );
}

export default function CodeBlockString({
  children,
  className: blockClassName = '',
  metastring,
  title: titleProp,
  showLineNumbers: showLineNumbersProp,
  language: languageProp,
}: Props): JSX.Element {
  const {
    prism: {defaultLanguage, magicComments},
  } = useThemeConfig();
  const language = normalizeLanguage(
    languageProp ?? parseLanguage(blockClassName) ?? defaultLanguage,
  );

  const prismTheme = usePrismTheme();
  const wordWrap = useCodeWordWrap();
  
  // Collapsible functionality
  const [isExpanded, setIsExpanded] = useState(false);
  const [isShort, setIsShort] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);

  // We still parse the metastring in case we want to support more syntax in the
  // future. Note that MDX doesn't strip quotes when parsing metastring:
  // "title=\"xyz\"" => title: "\"xyz\""
  const title = parseCodeBlockTitle(metastring) || titleProp;

  const {lineClassNames, code} = parseLines(children, {
    metastring,
    language,
    magicComments,
  });
  const showLineNumbers =
    showLineNumbersProp ?? containsLineNumbers(metastring);

  // Check if code block should be collapsible
  useEffect(() => {
    const lines = code.split('\n').length;
    const shouldCollapse = lines > 10;
    setIsShort(!shouldCollapse);
  }, [code]);

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <Container
      as="div"
      ref={containerRef}
      className={clsx(
        blockClassName,
        language &&
          !blockClassName.includes(`language-${language}`) &&
          `language-${language}`,
        isExpanded && 'expanded',
      )}
      data-short={isShort}>
      {title && <div className={styles.codeBlockTitle}>{title}</div>}
      <div className={styles.codeBlockContent}>
        <Highlight
          theme={prismTheme}
          code={code}
          language={(language ?? 'text') as Language}>
          {({className, style, tokens, getLineProps, getTokenProps}) => (
            <pre
              /* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex */
              tabIndex={0}
              ref={wordWrap.codeBlockRef}
              className={clsx(
                className, 
                styles.codeBlock, 
                'thin-scrollbar',
                isExpanded && 'expanded'
              )}
              style={style}>
              <code
                className={clsx(
                  styles.codeBlockLines,
                  showLineNumbers && styles.codeBlockLinesWithNumbering,
                )}>
                {tokens.map((line, i) => (
                  <Line
                    key={i}
                    line={line}
                    getLineProps={getLineProps}
                    getTokenProps={getTokenProps}
                    classNames={lineClassNames[i]}
                    showLineNumbers={showLineNumbers}
                  />
                ))}
              </code>
            </pre>
          )}
        </Highlight>
        <div className={styles.buttonGroup}>
          {(wordWrap.isEnabled || wordWrap.isCodeScrollable) && (
            <WordWrapButton
              className={styles.codeButton}
              onClick={() => wordWrap.toggle()}
              isEnabled={wordWrap.isEnabled}
            />
          )}
          <CopyButton className={styles.codeButton} code={code} />
        </div>
        <ExpandButton 
          isExpanded={isExpanded} 
          onClick={toggleExpanded} 
          isShort={isShort} 
        />
      </div>
    </Container>
  );
}
