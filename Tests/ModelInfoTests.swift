import XCTest

final class ModelInfoTests: XCTestCase {
    func testEveryModelHasInfoAndAPage() {
        for id in ModelID.allCases {
            XCTAssertFalse(id.info.usedFor.isEmpty, "\(id)")
            XCTAssertFalse(id.info.license.isEmpty, "\(id)")
            XCTAssertEqual(ModelCatalog.model(id).pageURL.host, "huggingface.co")
        }
    }

    func testEmptySearchListsEverythingInCatalogOrder() {
        XCTAssertEqual(ModelCatalog.search(""), ModelCatalog.all.map(\.id))
    }

    func testSearchMatchesNameLicenseAndRepoAndNeedsEveryWord() {
        XCTAssertEqual(ModelCatalog.search("triposr"), [.triposr])
        XCTAssertEqual(ModelCatalog.search("apache"), [.depthEstimation])
        XCTAssertTrue(ModelCatalog.search("tencent hunyuan").contains(.shapeFull))
        XCTAssertEqual(ModelCatalog.search("zero123 relief"), [])                 // both words must match
        XCTAssertEqual(ModelCatalog.search("  SHAPE   full "), ModelCatalog.search("full shape"))
    }

    func testCategoryFilterPartitionsTheCatalog() {
        let parts = ModelCategory.allCases.flatMap { ModelCatalog.search("", category: $0) }
        XCTAssertEqual(Set(parts), Set(ModelID.allCases))
        XCTAssertEqual(parts.count, ModelID.allCases.count)
        XCTAssertTrue(ModelCatalog.search("", category: .shape).contains(.shapeFull))
        XCTAssertFalse(ModelCatalog.search("", category: .paint).contains(.shapeFull))
    }
}
