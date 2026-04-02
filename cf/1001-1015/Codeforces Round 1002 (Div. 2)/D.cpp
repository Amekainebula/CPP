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
//int a[305][305];
//void solve()
//{
//    int n;
//    cin >> n;
//    for (int i = 1; i <= n; i++)
//    {
//        for (int j = 1; j <= n; j++)
//        {
//            cin >> a[i][j];
//        }
//    }
//    int maxx = 0;
//    for (int i = 1; i <= n; i++)
//    {
//        int cnt = 0;
//        for (int j = n; j >= 1; j--)
//        {
//            if (a[i][j] == 1)
//            {
//                cnt++;
//                maxx = max(maxx, cnt);
//            }
//            else
//                break;
//        }
//    }
//    cout << min(maxx + 1, n) << endl;
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