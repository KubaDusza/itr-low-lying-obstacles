-- "# References" heading -> IEEEtran BibTeX bibliography from refs.bib
-- Manual numerals are stripped from headings ("I. Introduction" -> "Introduction")
function Header(b)
  local s = pandoc.utils.stringify(b)
  if s == "References" then
    return pandoc.RawBlock("latex", "\\bibliographystyle{IEEEtran}\n\\bibliography{refs}")
  end
  local stripped = s:gsub("^[IVX]+%.%s+", "")
  if stripped ~= s then b.content = pandoc.Inlines(stripped) end
  return b
end
