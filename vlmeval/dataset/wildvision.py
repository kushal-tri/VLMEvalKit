import re
from functools import partial

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
        model = judge_kwargs['model']
        storage = get_intermediate_file_path(eval_file, f'_{model}')
        score_file = get_intermediate_file_path(eval_file, f'_{model}_score', 'csv')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            raw_data = WildVision('WildVision').data
            b64_map = {x: y for x, y in zip(raw_data['index'], raw_data['image'])}
            data = self.gen_eval_base(eval_file, b64_map)

            judge_kwargs['system_prompt'] = SYSTEM_PROMPT
            judge_kwargs['temperature'] = 0
            judge_kwargs['img_detail'] = 'high'
            judge_kwargs['timeout'] = 300
            model = build_judge(max_tokens=4096, **judge_kwargs)

            assert model.working(), ('WildVision evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    WildVision_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    ans[k] = v

            data['score'] = [ans[idx] for idx in data['index']]
            data.pop('image')
            dump(data, storage)

        data = load(storage)
        lt = len(data)

        scores = defaultdict(lambda: 0)
        for i in range(lt):
            item = data.iloc[i]
            if item['score'] not in self.score_map:
                score = 0
            else:
                score = self.score_map[item['score']]
                if '_rev' in item['index']:
                    score = -score
            scores[score] += 1
        name_map = {
            2: 'Much Better',
            1: 'Better',
            0: 'Tie',
            -1: 'Worse',
            -2: 'Much Worse'
        }
        scores = {name_map[k]: v for k, v in scores.items()}
        much_better = scores.get('Much Better', 0)
        better = scores.get('Better', 0)
        worse = scores.get('Worse', 0)
        much_worse = scores.get('Much Worse', 0)
        scores['Reward'] = (
            100 * much_better + 50 * better - 50 * worse - 100 * much_worse
        ) / lt
        scores['Win Rate'] = (better + much_better) / lt
        scores = {k: [v] for k, v in scores.items()}
        scores = pd.DataFrame(scores)
        dump(scores, score_file)
        return scores
