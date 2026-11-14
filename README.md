# test-runs

CI runs for work on [JuliaBUGS.jl](https://github.com/TuringLang/JuliaBUGS.jl) and [mcmcjs](https://github.com/mcmcjs/mcmcjs): test suites and BUGS example fits that are too heavy for a laptop.

Code under test is pushed here as branches, `juliabugs/<name>` and `mcmcjs/<name>`, and the workflows on `main` are started by hand with the branch as input:

```sh
gh workflow run juliabugs-tests.yml -R shravanngoswamii/test-runs -f ref=juliabugs/<name>
gh workflow run juliabugs-docs.yml -R shravanngoswamii/test-runs -f ref=juliabugs/<name>
gh workflow run mcmcjs-tests.yml -R shravanngoswamii/test-runs -f ref=mcmcjs/<name>
gh workflow run bugs-fits.yml -R shravanngoswamii/test-runs -f mcmcjs_ref=mcmcjs/<name> -f juliabugs_ref=juliabugs/<name> -f examples='["endo"]'
```

The pushed branches carry their projects' own workflows, but those only run on `main` or on pull requests, so nothing else starts here.
