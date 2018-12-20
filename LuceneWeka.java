
import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import java.io.File;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomTree;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

//This project is used to test the learning effect
public class LuceneWeka {

	static String indexPath = "./index/";
	static String dir_testfile = "./data/newfile.arff/";
	static String pathname = "./data/iris.arff";
	
	public static void main(String args[]) throws Exception {
		Analyzer analyzer = new StandardAnalyzer();
		Directory directory = FSDirectory.open(Paths.get(indexPath));

		ReadAndIndex(analyzer, directory);
		WriteNewFile(analyzer, directory);
		WekaTest();
	}

	public static void ReadAndIndex(Analyzer analyzer, Directory directory) 
			throws Exception {
		// To store an index on disk, use this instead:
		IndexWriterConfig config = new IndexWriterConfig(analyzer);
		IndexWriter iwriter = new IndexWriter(directory, config);
		
		FileReader reader = new FileReader(pathname);
		BufferedReader br = new BufferedReader(reader);
		String line;
		while ((line = br.readLine()) != null) {
			//
			if (line.equals("") || line.startsWith("%") || line.startsWith("@"))
				continue;

			String[] str_line = line.split(",");
			Document doc = new Document();
			doc.add(new Field("f1", str_line[0], TextField.TYPE_STORED));
			doc.add(new Field("f2", str_line[1], TextField.TYPE_STORED));
			doc.add(new Field("f3", str_line[2], TextField.TYPE_STORED));
			doc.add(new Field("f4", str_line[3], TextField.TYPE_STORED));
			doc.add(new Field("class", str_line[4], TextField.TYPE_STORED));
			iwriter.addDocument(doc);
			// System.out.println(str_line[0]);
		}
		br.close();
		iwriter.close();
	}

	public static void WriteNewFile(Analyzer analyzer, Directory directory)
			throws Exception {

		ArrayList<String> classList = new ArrayList<>();
		classList.add("setosa");
		classList.add("versicolor");

		FileWriter writer = null;
		writer = new FileWriter(dir_testfile, false);
		writer.write("@RELATION iris\n");
		writer.write("@ATTRIBUTE sepallength	REAL\n");
		writer.write("@ATTRIBUTE sepalwidth	REAL\n");
		writer.write("@ATTRIBUTE petallength	REAL\n");
		writer.write("@ATTRIBUTE class 	{Iris-setosa,Iris-versicolor}\n");
		writer.write("@DATA\n");

		// Now search the index:
		DirectoryReader ireader = DirectoryReader.open(directory);
		IndexSearcher isearcher = new IndexSearcher(ireader);
		QueryParser parser = new QueryParser("class", analyzer);

		for (String keyword : classList) {
			Query query = parser.parse(keyword);
			ScoreDoc[] hits = isearcher.search(query, 1000).scoreDocs;
			if (hits.length > 0) {
				for (ScoreDoc hit : hits) {
					Document hitDoc = isearcher.doc(hit.doc);
					String temp = hitDoc.get("f2") + "," + hitDoc.get("f3") + "," + 
									hitDoc.get("f4") + ","+ hitDoc.get("class");
					writer.write(temp + "\n");
				}
			}
		}
		ireader.close();
		writer.close();

	}

	// weka十折交叉验证
	public static void WekaTest() throws Exception {
		

		Classifier m_classifier = new RandomTree();//
		
		File inputFile = new File(dir_testfile);//
		ArffLoader atf = new ArffLoader(); //
		atf.setFile(inputFile);
		Instances instancesTrain = atf.getDataSet(); //
		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
		m_classifier.buildClassifier(instancesTrain);

		Evaluation eval = new Evaluation(instancesTrain);
		eval.crossValidateModel(m_classifier, instancesTrain, 10, new Random(1));
		System.out.print(eval.toSummaryString());
		System.out.print(eval.toMatrixString());
		System.out.print(eval.toClassDetailsString());
	}

}
