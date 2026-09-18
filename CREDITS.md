# Credits and licences

**This repository (code, `EXPERIMENTS.md`, the circuit builder and the analysis scripts): MIT**, see
`LICENSE`. Everything below came from somewhere else.

## Connectome data — CC-BY 4.0

`mk/fly_circuit.js` is derived from the **maleCNS v1.0** connectome: a subgraph of 12,000 neurons
with their synapse counts, transmitter predictions and annotations, produced by
`scripts/build_malecns_fighter.py` from the public flat-connectome files.

> maleCNS v1.0, FlyEM (HHMI Janelia) with the University of Cambridge, the MRC Laboratory of
> Molecular Biology and Google Research. <https://male-cns.janelia.org/> — licensed CC-BY 4.0.

## Game engine — MIT

**[mk.js](https://github.com/mgechev/mk.js)** by Minko Gechev, MIT licensed, vendored at
`mk/vendor/mk.js` with two changes marked in the source (a duplicate sprite preload removed; the
hit test left intact while the AI reads distance from the collision box).

## Game art — not redistributed

mk.js ships Mortal Kombat character and arena sprites, which belong to their rights holders
(Midway / NetherRealm) and are **not** in this repository. `scripts/fetch_game_assets.py` downloads
them from the upstream mk.js repository into `mk/images/` for local play.

## Sound effects — not redistributed

Punch, kick and reaction one-shots come from Pixabay, whose licence does not allow redistributing
the files as standalone assets. Download them yourself, drop them in `mk/effects/`, and run
`python -m scripts.slice_effects`. The ones used here:

- Punches and hits — [storegraphic](https://pixabay.com/users/storegraphic-49061086/) on [Pixabay](https://pixabay.com/) (content 310521)
- Kick, bright medium — [Kho âm Thanh](https://pixabay.com/users/khoamthanh-48236707/) on Pixabay (content 504170)
- "Umph" reaction — [freesound_community](https://pixabay.com/users/freesound_community-46691455/) on Pixabay (content 47201)
- "Ough" reaction — freesound_community on Pixabay (content 47202)
- Kick pack — freesound_community on Pixabay (content 38706)

Without them the game synthesizes its hits instead; music, blocks, whooshes, the bell and the KO
boom are synthesized either way (`mk/audio.js`, no samples).

## Announcer — generate your own

The announcer lines were generated with **ElevenLabs** and are not included. With an
`ELEVENLABS_API_KEY` in `.env`, `python -m scripts.bake_announcer` bakes them into
`mk/voice/<set>/`. Without them the game runs silently announced.

## Fonts — SIL Open Font License 1.1

**Metal Mania** and **Teko** from Google Fonts, both OFL 1.1, at `mk/fonts/`.

## Jev

**[TypeSafe](https://typesafe.ai/)**'s System One model `jev-latest`, called through `jev.py` with
your own `TYPESAFE_API_KEY`. `?jev=local` and `?jev=rules` need no key and no account.
