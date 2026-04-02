////#define _CRT_SECURE_NO_WARNINGS 1
////#include <bits/stdc++.h>
////#define ll long long
////#define ld long double
////#define ull unsigned long long
////#define endl '\n'
////using namespace std;
////map<string, int> mp;
////void solve()
////{
////    int n;
////    cin >> n;
////}
////int main()
////{
////    ios::sync_with_stdio(false);
////    cin.tie(0);
////    ll T = 1;
////    cin >> T;
////    while (T--)
////    {
////        solve();
////    }
////    return 0;
////}
//#include <bits/stdc++.h>
//using namespace std;
//string mp[1005];
//pair<int, int> ans[1005];
//
//int main()
//{
//    int t;
//    cin >> t;
//    while (t--)
//    {
//        int n;
//        cin >> n;
//        for (int i = 1; i <= n; i++)
//        {
//            cin >> mp[i];
//        }
//        for (int i = 1; i <= n; i++)
//        {
//            ans[i].first = 0;
//            ans[i].second = i;
//        }
//        for (int i = n; i >= 1; i--)
//        {
//            int dx = 0;
//            for (int j = n - 1; j >= i; j--)
//            {
//                if (mp[i][j] == '1') {
//                    ans[j + 1].first++;
//                }
//                else dx++;
//            }
//            ans[i].first = 1 + dx;
//        }
//        sort(ans + 1, ans + 1 + n);
//        for (int i = 1; i <= n; i++)
//        {
//            cout << ans[i].second << " ";
//        }
//        cout << endl;
//    }
//    return 0;
//}
#include <bits/stdc++.h>
using namespace std;
char mp[1005][1005];
int b[1005];
void solve()
{
    int n;
    cin >> n;
    for (int i = 1; i <= n; i++)
    {
        int q = 0;
        for (int j = 1; j <= n; j++)
        {
            cin >> mp[i][j];
            if ((i < j && mp[i][j] == '0') || (i > j && mp[i][j] == '1'))
                q++;
        }
        b[q + 1] = i;
    }
    for (int i = 1; i <= n; i++)
        cout << b[i] << " ";
    cout << endl;
}
int main()
{
    int t;
    cin >> t;
    while (t--)
    {
        solve();
    }
    return 0;
}