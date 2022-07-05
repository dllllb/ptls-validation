import luigi
import hydra

from omegaconf import DictConfig, OmegaConf
from embeddings_validation import ReportCollect
from embeddings_validation.config import Config
from embeddings_validation.tasks.fold_splitter import FoldSplitter


@hydra.main(version_base=None)
def main(conf: DictConfig):
    conf.workers = conf.get('workers')
    if conf.workers is None: raise AttributeError('Define the number of workers: +workers=4')
    conf.total_cpu_count = conf.get('total_cpu_count')
    if conf.total_cpu_count is None: raise AttributeError('Define the number of cpu on your machine: +total_cpu_count=8')

    config = Config.get_conf(conf, conf.conf_path)

    if conf.get('split_only', False):
        task = FoldSplitter(
            conf=config,
        )
    else:
        task = ReportCollect(
            conf=config,
            total_cpu_count=conf['total_cpu_count'],
        )
    luigi.build([task], workers=conf['workers'],
                        local_scheduler=conf.get('local_scheduler', True),
                        log_level=conf.get('log_level', 'INFO'))


if __name__ == '__main__':
    main()

