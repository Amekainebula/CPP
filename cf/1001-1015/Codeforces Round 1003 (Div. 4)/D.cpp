//#define _CRT_SECURE_NO_WARNINGS 1
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
//using namespace std;
//bool cmp(pii a, pii b)
//{
//    return a.se > b.se;
//}
//void solve()
//{
//    int n, m;
//    cin >> n >> m;
//    vector<vector<int>> a(n, vector<int>(m));
//    vector<pii>deta;
//    for (int i = 0; i < n; i++)
//    {
//        int sum = 0;
//        for (int j = 0; j < m; j++)
//        {
//            cin >> a[i][j];
//            sum += a[i][j];
//        }
//        deta.pb(mpr(i, sum));
//    }
//    sort(all(deta), cmp);
//    int ans_sum = 0;
//    int cnt = n * m;
//    for (int i = 0; i < n; i++)
//    {
//        int temp = deta[i].fi;
//        for (int j = 0; j < m; j++)
//        {
//            ans_sum += a[temp][j] * cnt;
//            cnt--;
//        }
//    }
//    cout << ans_sum << endl;
//}
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