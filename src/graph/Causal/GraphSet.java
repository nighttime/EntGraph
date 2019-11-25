package graph.Causal;

import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;
import constants.ConstantsGraphs;
import graph.PGraph;

import java.io.File;
import java.util.Arrays;
import java.util.Iterator;

public class GraphSet implements Iterable<PGraph> {
    private File[] files;

    {
        files = getGraphFiles();
    }

    public static GraphSet generator() {
        return new GraphSet();
    }

    private File[] getGraphFiles() {
        File folder = new File(ConstantsGraphs.root);
        File[] files = folder.listFiles();
        assert files != null;
        Arrays.sort(files);
        return files;
    }

    @Override
    public Iterator<PGraph> iterator() {
        return new GraphIter();
    }

    private class GraphIter implements Iterator<PGraph> {
        private PeekingIterator<File> x;

        {
            x = Iterators.peekingIterator(Arrays.asList(files).iterator());
        }

        @Override
        public boolean hasNext() {
            while (x.hasNext() && !x.peek().getName().contains(ConstantsGraphs.suffix)) {
                x.next();
            }
            return x.hasNext();
        }

        @Override
        public PGraph next() {
            return new PGraph(ConstantsGraphs.root + x.next().getName());
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("remove not supported");
        }
    }
}
