//#include <bits/stdc++.h>
//#define int long long
//#define ld long double
//#define ull unsigned long long
//#define lowbit(x) (x & -x)
//#define pb push_back
//#define pii pair<int, int>
//#define mpr make_pair
//#define fi first
//#define se second
//#define all(x) x.begin(), x.end()
//#define rall(x) x.rbegin(), x.rend()
//#define sz(x) (int)(x).size()
//#define INF 0x7fffffffffffffff
//#define endl '\n' 
//#define yes cout << "YES"<< endl
//#define no cout << "NO"<< endl
//#define yess cout << "Yes" << endl
//#define noo cout << "No"<< endl
//using namespace std;
//int a[2005];
//struct node
//{
//    int r, maxx;
//}arr[2005];
//
//void solve() {
//    int n;
//    cin >> n;
//    vector<int>a(n);
//    for (int i = 0; i < n; i++) {
//        cin >> a[i];
//    }
//
//    vector<pair<int, int>>ans(n);
//    for (int i = 0; i < n; i++) {
//        int cnt1 = 0, cnt2 = 0, mx = 0, r = i;
//        for (int j = i; j < n; j++) {
//            if (a[j] > a[i])cnt1++;
//            else if (a[j] < a[i]) cnt2++;
//            if (cnt2 - cnt1 > mx) {
//                mx = cnt2 - cnt1;
//                r = j;
//            }
//        }
//        ans[i] = make_pair(mx, r);
//    }
//    int l = 1, r = 1, mx = 0;
//    for (int i = 0; i < n; i++) {
//        auto& [x, y] = ans[i];
//        //		cout<<x<<" "<<y<<endl;
//        if (ans[i].first > mx) {
//            l = i + 1, r = ans[i].second + 1;
//            mx = ans[i].first;
//        }
//    }
//    cout << l << " " << r << endl;
//    //	cout<<endl;
//}
////void solve()
////{
////    int n;
////    cin >> n;
////    for (int i = 1; i <= n; i++)
////        cin >> a[i];
////    for (int i = 1; i <= n; i++)
////    {
////        int cntmax = 0, cntmin = 0,mx = 0,r = i;
////        for (int j = i + 1; j <= n; j++)
////        {
////           if (a[j] > a[i])cntmax++;
////           else if (a[j] < a[i]) cntmin++;
////           if (cntmax - cntmin > mx)
////           {
////               mx = cntmax - cntmin;
////               r = j;
////           }
////        }
////        arr[i] = {r, mx};
////    }
////    int l = 1, r = 1, mx = 0;
////    for (int i = 1; i <= n; i++)
////    {
////        if (arr[i].maxx > mx)
////        {
////            l = i, r = arr[i].r, mx = arr[i].maxx;
////        }
////    }
////    cout << l << " " << r << endl;
////}                               
//signed main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    cout.tie(0);
//    int T = 1;
//    cin >> T;
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}