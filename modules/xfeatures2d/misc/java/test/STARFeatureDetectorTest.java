package org.opencv.test.features;

import java.util.Arrays;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.KeyPoint;
import org.opencv.test.OpenCVTestCase;
import org.opencv.test.OpenCVTestRunner;
import org.opencv.imgproc.Imgproc;
import org.opencv.xfeatures2d.StarDetector;

public class STARFeatureDetectorTest extends OpenCVTestCase {

    StarDetector detector;
    int matSize;
    KeyPoint[] truth;

    private Mat getMaskImg() {
        Mat mask = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Mat right = mask.submat(0, matSize, matSize / 2, matSize);
        right.setTo(new Scalar(0));
        return mask;
    }

    private Mat getTestImg() {
        Scalar color = new Scalar(0);
        int center = matSize / 2;
        int radius = 6;
        int offset = 40;

        Mat img = new Mat(matSize, matSize, CvType.CV_8U, new Scalar(255));
        Imgproc.circle(img, new Point(center - offset, center), radius, color, -1);
        Imgproc.circle(img, new Point(center + offset, center), radius, color, -1);
        Imgproc.circle(img, new Point(center, center - offset), radius, color, -1);
        Imgproc.circle(img, new Point(center, center + offset), radius, color, -1);
        Imgproc.circle(img, new Point(center, center), radius, color, -1);
        return img;
    }

    protected void setUp() throws Exception {
        super.setUp();
        detector = createClassInstance(XFEATURES2D+"StarDetector", DEFAULT_FACTORY, null, null);
        matSize = 200;
        truth = new KeyPoint[] {
                new KeyPoint( 95,  80, 22, -1, 31.5957f, 0, -1),
                new KeyPoint(105,  80, 22, -1, 31.5957f, 0, -1),
                new KeyPoint( 80,  95, 22, -1, 31.5957f, 0, -1),
                new KeyPoint(120,  95, 22, -1, 31.5957f, 0, -1),
                new KeyPoint(100, 100,  8, -1, 30.f,     0, -1),
                new KeyPoint( 80, 105, 22, -1, 31.5957f, 0, -1),
                new KeyPoint(120, 105, 22, -1, 31.5957f, 0, -1),
                new KeyPoint( 95, 120, 22, -1, 31.5957f, 0, -1),
                new KeyPoint(105, 120, 22, -1, 31.5957f, 0, -1)
            };
    }

    public void testCreate() {
        assertNotNull(detector);
    }

    public void testDetectListOfMatListOfListOfKeyPoint() {
        fail("Not yet implemented");
    }

    public void testDetectListOfMatListOfListOfKeyPointListOfMat() {
        fail("Not yet implemented");
    }

    public void testDetectMatListOfKeyPoint() {
        Mat img = getTestImg();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();

        detector.detect(img, keypoints);

        assertListKeyPointEquals(Arrays.asList(truth), keypoints.toList(), EPS);
    }

    public void testDetectMatListOfKeyPointMat() {
        Mat img = getTestImg();
        Mat mask = getMaskImg();
        MatOfKeyPoint keypoints = new MatOfKeyPoint();

        detector.detect(img, keypoints, mask);

        assertListKeyPointEquals(Arrays.asList(truth[0], truth[2], truth[5], truth[7]), keypoints.toList(), EPS);
    }

    public void testEmpty() {
//        assertFalse(detector.empty());
        fail("Not yet implemented");
    }

    public void testReadYml() {
        Mat img = getTestImg();

        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        detector.detect(img, keypoints1);

        String filename = OpenCVTestRunner.getTempFileName("yml");
        writeFile(filename, "%YAML:1.0\n---\nname: \"Feature2D.STAR\"\nmaxSize: 45\nresponseThreshold: 150\nlineThresholdProjected: 10\nlineThresholdBinarized: 8\nsuppressNonmaxSize: 5\n");
        detector.read(filename);

        assertEquals(45, detector.getMaxSize());
        assertEquals(150, detector.getResponseThreshold());
        assertEquals(10, detector.getLineThresholdProjected());
        assertEquals(8, detector.getLineThresholdBinarized());
        assertEquals(5, detector.getSuppressNonmaxSize());

        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        detector.detect(img, keypoints2);

        assertTrue(keypoints2.total() <= keypoints1.total());
    }

    public void testWriteYml() {
        String filename = OpenCVTestRunner.getTempFileName("yml");

        detector.write(filename);

        String truth = "%YAML:1.0\n---\nname: \"Feature2D.STAR\"\nmaxSize: 45\nresponseThreshold: 30\nlineThresholdProjected: 10\nlineThresholdBinarized: 8\nsuppressNonmaxSize: 5\n";
        assertEquals(truth, readFile(filename));
    }

}
